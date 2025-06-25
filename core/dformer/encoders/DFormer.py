import warnings
from collections import OrderedDict
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout

# from mmcv.cnn.utils.weight_init import (constant_init, trunc_normal_,
#                                         trunc_normal_init)
# from mmcv.runner import (BaseModule, CheckpointLoader, ModuleList,
#                          load_state_dict)
from mmengine.model.base_module import BaseModule
from mmengine.runner.checkpoint import load_state_dict

# from mmcv.utils import to_2tuple
import math


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, norm_cfg=dict(type="SyncBN", requires_grad=True)):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pos(x) + x
        x = x.permute(0, 2, 3, 1)
        x = self.act(x)
        x = self.fc2(x)

        return x


class attention(nn.Module):
    def __init__(self, dim, num_head=8, window=7, norm_cfg=dict(type="SyncBN", requires_grad=True), drop_sm=False):
        # dim: the input feature dimension
        # num_head: number of attention heads for multi-head attention
        # window: the size of the local window used in attention computation
        # norm_cfg: configuration dictionary for normalization layers (not used directly in this class)
        # drop_sm: if True, disables SM-specific operations (used to drop SM stream)

        super().__init__()
        self.num_head = num_head  # Number of attention heads
        self.window = window  # Local window size for attention

        self.q = nn.Linear(dim, dim)  # Linear layer to generate query from input
        self.q_cut = nn.Linear(dim, dim // 2)  # Linear layer to produce reduced query for SM stream
        self.a = nn.Linear(dim, dim)  # Linear layer to project attention feature map
        self.l = nn.Linear(dim, dim)  # Linear layer before applying convolution

        self.conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)  # Depth-wise conv for attention map
        self.e_conv = nn.Conv2d(dim // 2, dim // 2, 7, padding=3, groups=dim // 2)  # Depth-wise conv for SM stream
        # self.e_fore = nn.Linear(dim // 2, dim // 2)  # Foreground linear projection for depth stream
        # self.e_back = nn.Linear(dim // 2, dim // 2)  # Background linear projection for depth stream

        self.proj = nn.Linear(dim // 2 * 3, dim)  # Final projection after concatenating attention outputs
        if not drop_sm:
            self.proj_e = nn.Linear(dim // 2 * 3, dim // 2)  # Projection for SM stream if not dropped

        if window != 0:
            self.short_cut_linear = nn.Linear(dim // 2 * 3, dim // 2)  # Linear layer for shortcut path
            self.kv = nn.Linear(dim, dim)  # Linear layer to generate keys and values
            self.pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))  # Spatial pooling for windowed attention
            self.proj = nn.Linear(dim * 2, dim)  # Adjust projection when windowed attention is used
            if not drop_sm:
                self.proj_e = nn.Linear(dim * 2, dim // 2)  # Adjust SM projection

        self.act = nn.GELU()  # Activation function
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")  # Normalization for Depth input
        self.norm_e = LayerNorm(dim // 2, eps=1e-6, data_format="channels_last")  # Normalization for SM input
        self.drop_sm = drop_sm  # Flag to drop depth stream

    def forward(self, x, x_e):
        B, H, W, C = x.size()  # Batch size, height, width, channel
        x = self.norm(x)  # Normalize Depth input
        x_e = self.norm_e(x_e)  # Normalize SM input

        if self.window != 0:
            short_cut = torch.cat([x, x_e], dim=3)  # Concatenate Depth and SM
            short_cut = short_cut.permute(0, 3, 1, 2)  # Convert to NCHW for pooling

        q = self.q(x)  # Compute query from Depth
        cutted_x = self.q_cut(x)  # Reduced query for fusion
        x = self.l(x).permute(0, 3, 1, 2)  # Apply linear layer and reshape for conv
        x = self.act(x)  # Activate

        a = self.conv(x)  # Apply attention convolution
        a = a.permute(0, 2, 3, 1)  # Convert back to NHWC
        a = self.a(a)  # Project attention map

        if self.window != 0:
            b = x.permute(0, 2, 3, 1)  # Prepare for key-value projection
            kv = self.kv(b)  # Compute key and value
            kv = kv.reshape(B, H * W, 2, self.num_head, C // self.num_head // 2).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)  # Split into keys and values
            short_cut = self.pool(short_cut).permute(0, 2, 3, 1)  # Pool shortcut feature
            short_cut = self.short_cut_linear(short_cut)  # Project shortcut
            short_cut = short_cut.reshape(B, -1, self.num_head, C // self.num_head // 2).permute(0, 2, 1, 3)
            m = short_cut  # Use shortcut as query
            attn = (m * (C // self.num_head // 2) ** -0.5) @ k.transpose(-2, -1)  # Attention scores
            attn = attn.softmax(dim=-1)  # Softmax over keys
            attn = (
                (attn @ v)
                .reshape(B, self.num_head, self.window, self.window, C // self.num_head // 2)
                .permute(0, 1, 4, 2, 3)
                .reshape(B, C // 2, self.window, self.window)
            )
            attn = F.interpolate(attn, (H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)  # Resize

        x_e = self.e_conv(x_e.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # Refined SM
        cutted_x = cutted_x * x_e  # Fuse Depth and SM with elementwise product
        x = q * a  # Combine query and attention

        if self.window != 0:
            x = torch.cat([x, attn, cutted_x], dim=3)  # Concatenate Depth attention, SM attention, and fused features
        else:
            x = torch.cat([x, cutted_x], dim=3)  # Only use fused features if no windowed attention

        if not self.drop_sm:
            x_e = self.proj_e(x)  # Project features for SM stream

        x = self.proj(x)  # Final projection

        return x, x_e  # Return updated Depth and SM features


class Block(nn.Module):
    def __init__(
        self,
        index,  # Index of this block in the overall network
        dim,  # Feature dimension of the input and output
        num_head,  # Number of attention heads
        norm_cfg=dict(type="SyncBN", requires_grad=True),  # Normalization config (not used directly here)
        mlp_ratio=4.0,  # Expansion ratio for the hidden layer in the MLP
        block_index=0,  # Index of this block within its stage
        last_block_index=50,  # Last block index for this stage (used to control window usage)
        window=7,  # Size of the local attention window
        dropout_layer=None,  # Optional dropout layer configuration
        drop_sm=False,  # Whether to drop the SM stream in this block
    ):
        super().__init__()

        self.index = index  # Save index of this block
        layer_scale_init_value = 1e-6  # Initial value for layer scale

        if block_index > last_block_index:
            window = 0  # Disable local attention if block is beyond last_block_index

        self.attn = attention(dim, num_head, window=window, norm_cfg=norm_cfg, drop_sm=drop_sm)  # Attention module
        self.mlp = MLP(dim, mlp_ratio, norm_cfg=norm_cfg)  # MLP for depth stream
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else torch.nn.Identity()  # Optional dropout

        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)  # Scaling factor for attention residual
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)  # Scaling factor for MLP residual

        if not drop_sm:
            self.layer_scale_1_e = nn.Parameter(layer_scale_init_value * torch.ones((dim // 2)), requires_grad=True)  # Scale for depth attention residual
            self.layer_scale_2_e = nn.Parameter(layer_scale_init_value * torch.ones((dim // 2)), requires_grad=True)  # Scale for depth MLP residual
            self.mlp_e2 = MLP(dim // 2, mlp_ratio)  # MLP for SM stream

        self.drop_sm = drop_sm  # Save drop_sm flag

    def forward(self, x, x_e):
        res_x, res_e = x, x_e  # Save residuals
        x, x_e = self.attn(x, x_e)  # Apply attention to both depth and SM streams

        # Residual connection with layer scale and optional dropout for depth
        x = res_x + self.dropout_layer(self.layer_scale_1.unsqueeze(0).unsqueeze(0) * x)
        x = x + self.dropout_layer(self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(x))

        if not self.drop_sm:
            # Residual connection with layer scale and optional dropout for SM
            x_e = res_e + self.dropout_layer(self.layer_scale_1_e.unsqueeze(0).unsqueeze(0) * x_e)
            x_e = x_e + self.dropout_layer(self.layer_scale_2_e.unsqueeze(0).unsqueeze(0) * self.mlp_e2(x_e))

        return x, x_e  # Return updated depth and SM features


class DFormer(BaseModule):
    def __init__(
        self,
        in_channels=2,  # Number of input channels
        depths=(2, 2, 8, 2),  # Number of blocks in each stage
        dims=(32, 64, 128, 256),  # Feature dimensions for each stage
        out_indices=(0, 1, 2, 3),  # Indices of stages whose outputs are returned
        windows=[7, 7, 7, 7],  # Window sizes for local attention in each stage
        norm_cfg=dict(type="SyncBN", requires_grad=True),  # Normalization config for convolution layers
        mlp_ratios=[8, 8, 4, 4],  # MLP expansion ratios for each stage
        num_heads=(2, 4, 10, 16),  # Number of attention heads per stage
        last_block=[50, 50, 50, 50],  # Last block index for controlling window usage in each stage
        drop_path_rate=0.1,  # Drop path rate for regularization
        init_cfg=None,  # Initialization configuration (e.g., pretrained weights)
    ):
        super().__init__()
        print(drop_path_rate)
        self.depths = depths  # Save the number of blocks per stage
        self.init_cfg = init_cfg  # Save initialization config
        self.out_indices = out_indices  # Save output stage indices

        self.downsample_layers = nn.ModuleList()  # Layers to downsample Depth input
        stem = nn.Sequential(  # Stem for Depth input
            nn.Conv2d(1, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0] // 2),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0]),
        )

        self.downsample_layers_e = nn.ModuleList()  # Layers to downsample SM input
        stem_e = nn.Sequential(  # Stem for SM input
            nn.Conv2d(1, dims[0] // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0] // 4),
            nn.GELU(),
            nn.Conv2d(dims[0] // 4, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0] // 2),
        )

        self.downsample_layers.append(stem)  # Add RGB stem to downsample layers
        self.downsample_layers_e.append(stem_e)  # Add depth stem to downsample layers

        for i in range(len(dims) - 1):  # Create downsample layers between stages
            stride = 2
            downsample_layer = nn.Sequential(
                build_norm_layer(norm_cfg, dims[i])[1],
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=stride, padding=1),
            )
            self.downsample_layers.append(downsample_layer)

            downsample_layer_e = nn.Sequential(
                build_norm_layer(norm_cfg, dims[i] // 2)[1],
                nn.Conv2d(dims[i] // 2, dims[i + 1] // 2, kernel_size=3, stride=stride, padding=1),
            )
            self.downsample_layers_e.append(downsample_layer_e)

        self.stages = nn.ModuleList()  # Initialize stage container
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # Drop path rates per block
        cur = 0  # Track block index across stages

        for i in range(len(dims)):  # Construct stages with blocks
            stage = nn.Sequential(
                *[
                    Block(
                        index=cur + j,
                        dim=dims[i],
                        window=windows[i],
                        dropout_layer=dict(type="DropPath", drop_prob=dp_rates[cur + j]),
                        num_head=num_heads[i],
                        norm_cfg=norm_cfg,
                        block_index=depths[i] - j,
                        last_block_index=last_block[i],
                        mlp_ratio=mlp_ratios[i],
                        drop_sm=((i == 3) & (j == depths[i] - 1)),  # Drop depth on last block of last stage
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

    def forward(self, x, x_e):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension if needed
        if len(x_e.shape) == 3:
            x_e = x_e.unsqueeze(0)

        x_e = x_e[:, 0, :, :].unsqueeze(1)  # Use first channel for depth

        outs = []  # Output features from selected stages
        for i in range(4):
            x = self.downsample_layers[i](x)
            x_e = self.downsample_layers_e[i](x_e)

            x = x.permute(0, 2, 3, 1)  # Convert to NHWC format
            x_e = x_e.permute(0, 2, 3, 1)

            for blk in self.stages[i]:
                x, x_e = blk(x, x_e)  # Pass through blocks

            x = x.permute(0, 3, 1, 2)  # Convert back to NCHW
            x_e = x_e.permute(0, 3, 1, 2)
            outs.append(x)  # Save output

        return outs, None  # Return list of features and placeholder


def DFormer_Tiny(**kwargs):  # 81.5
    model = DFormer(
        dims=[32, 64, 128, 256],
        mlp_ratios=[8, 8, 4, 4],
        depths=[3, 3, 5, 2],
        num_heads=[1, 2, 4, 8],
        windows=[0, 7, 7, 7],
        **kwargs,
    )

    return model


def DFormer_Small(**kwargs):  # 81.0
    model = DFormer(
        dims=[64, 128, 256, 512],
        mlp_ratios=[8, 8, 4, 4],
        depths=[2, 2, 4, 2],
        num_heads=[1, 2, 4, 8],
        windows=[0, 7, 7, 7],
        **kwargs,
    )
    return model


def DFormer_Base(drop_path_rate=0.1, **kwargs):  # 82.1
    model = DFormer(
        dims=[64, 128, 256, 512],
        mlp_ratios=[8, 8, 4, 4],
        depths=[3, 3, 12, 2],
        num_heads=[1, 2, 4, 8],
        windows=[0, 7, 7, 7],
        drop_path_rate=drop_path_rate,
        **kwargs,
    )

    return model


def DFormer_Large(drop_path_rate=0.1, **kwargs):  # 82.1
    model = DFormer(
        dims=[96, 192, 288, 576],
        mlp_ratios=[8, 8, 4, 4],
        depths=[3, 3, 12, 2],
        num_heads=[1, 2, 4, 8],
        windows=[0, 7, 7, 7],
        drop_path_rate=drop_path_rate,
        **kwargs,
    )
    return model

def main():
    # Dummy input: batch size = 1, 2 channels (1 SM, 1 Depth), 224x224 resolution
    B, C, H, W = 1, 2, 224, 224
    dummy_input = torch.randn(B, C, H, W)

    rgb = dummy_input[:, 0, :, :].unsqueeze(1)  # SM input (1 channel1)
    depth = dummy_input[:, 1, :, :].unsqueeze(1)  # Depth input (1 channel)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Khởi tạo model
    model = DFormer_Tiny()
    model.to(device)
    model.eval()

    rgb = rgb.to(device)
    depth = depth.to(device)

    # Chạy forward
    with torch.no_grad():
        outputs, _ = model(rgb, depth)

    # In kích thước output của từng stage
    for i, out in enumerate(outputs):
        print(f"Output of stage {i}: shape = {out.shape}")

if __name__ == "__main__":
    main()