import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings

class EncoderDecoder(nn.Module):
    def __init__(
        self,
        cfg=None,  # Configuration object containing model options
        norm_layer=nn.BatchNorm2d,  # Normalization layer to use
        syncbn=False,  # Whether to use synchronized batch normalization
    ):
        super(EncoderDecoder, self).__init__()
        self.norm_layer = norm_layer
        self.cfg = cfg

        # Select backbone model and set channel dimensions accordingly
        if cfg.backbone == "DFormer-Tiny":
            from .encoders.DFormer import DFormer_Tiny as backbone
            self.channels = [32, 64, 128, 256]
        else:
            raise NotImplementedError  # Raise error if unknown backbone

        # Choose normalization config based on syncbn flag
        if syncbn:
            norm_cfg = dict(type="SyncBN", requires_grad=True)
        else:
            norm_cfg = dict(type="BN", requires_grad=True)

        # Instantiate backbone with or without drop path rate
        if cfg.drop_path_rate is not None:
            self.backbone = backbone(drop_path_rate=cfg.drop_path_rate, norm_cfg=norm_cfg)
        else:
            self.backbone = backbone(drop_path_rate=0.1, norm_cfg=norm_cfg)

        self.aux_head = None  # Initialize auxiliary head as None

        # Choose decoder type
        if cfg.decoder == "MLPDecoder":
            from .decoders.MLPDecoder import DecoderHead
            self.decode_head = DecoderHead(
                in_channels=self.channels,
                num_classes=cfg.num_classes,
                norm_layer=norm_layer,
                embed_dim=cfg.decoder_embed_dim,
            )

        elif cfg.decoder == "ham":
            from .decoders.ham_head import LightHamHead as DecoderHead
            self.decode_head = DecoderHead(
                in_channels=self.channels[1:],  # Use only later stages
                in_index=[1, 2, 3],
                norm_cfg=norm_cfg,
                channels=cfg.decoder_embed_dim,
            )
        else:
            from .decoders.fcnhead import FCNHead
            self.decode_head = FCNHead(
                in_channels=self.channels[-1],  # Use final stage output
                kernel_size=3,
                num_classes=cfg.num_classes,
                norm_layer=norm_layer,
            )

    def encode_decode(self, depth, sm):
        """Encode images with backbone and decode into a semantic segmentation map of the same size as input."""
        orisize = depth.shape  # Original input size for upsampling output
        x = self.backbone(depth, sm)  # Pass through encoder

        enc_out, _ = x

        for i, out in enumerate(enc_out):
            print(f"Output of stage {i}: shape = {out.shape}")

        out = self.decode_head.forward(enc_out)  # Run decoder
        out = F.interpolate(out, size=orisize[-2:], mode="bilinear", align_corners=False)  # Upsample to input size

        return out

    def forward(self, depth, sm):
        """Forward function. Optionally returns auxiliary output."""
        out = self.encode_decode(depth, sm)

        return out
    

class DummyCfg:
    def __init__(self):
        self.backbone = "DFormer-Tiny"
        self.decoder = "ham"
        self.decoder_embed_dim = 256
        self.drop_path_rate = 0.15

def main():
    # Dummy input: batch size = 1, 2 channels (1 SM, 1 Depth), 224x224 resolution
    B, C, H, W = 2, 2, 224, 224
    dummy_input = torch.randn(B, C, H, W)

    SM = dummy_input[:, 0, :, :].unsqueeze(1)   # (B, 1, H, W)
    depth = dummy_input[:, 1, :, :].unsqueeze(1) # (B, 1, H, W)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dummy cfg and model
    cfg = DummyCfg()
    model = EncoderDecoder(cfg=cfg, norm_layer=nn.BatchNorm2d, syncbn=False)

    model.to(device)
    model.eval()

    SM = SM.to(device)
    depth = depth.to(device)

    with torch.no_grad():
        outputs = model(depth, SM)

    # Handle output types
    if isinstance(outputs, tuple):
        main_out, aux_out = outputs
        print(f"Main output shape: {main_out.shape}")
        print(f"Auxiliary output shape: {aux_out.shape}")
    else:
        print(f"Output shape: {outputs.shape}")

if __name__ == "__main__":
    main()