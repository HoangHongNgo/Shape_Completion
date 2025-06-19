import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.init_func import init_weight
from utils.load_utils import load_pretrain
from functools import partial

from utils.engine.logger import get_logger
import warnings

logger = get_logger()

class EncoderDecoder(nn.Module):
    def __init__(
        self,
        cfg=None,
        norm_layer=nn.BatchNorm2d,
        syncbn=False,
    ):
        super(EncoderDecoder, self).__init__()
        self.norm_layer = norm_layer
        self.cfg = cfg

        if cfg.backbone == "DFormer-Large":
            from .encoders.DFormer import DFormer_Large as backbone

            self.channels = [96, 192, 288, 576]
        elif cfg.backbone == "DFormer-Base":
            from .encoders.DFormer import DFormer_Base as backbone

            self.channels = [64, 128, 256, 512]
        elif cfg.backbone == "DFormer-Small":
            from .encoders.DFormer import DFormer_Small as backbone

            self.channels = [64, 128, 256, 512]
        elif cfg.backbone == "DFormer-Tiny":
            from .encoders.DFormer import DFormer_Tiny as backbone

            self.channels = [32, 64, 128, 256]
        else:
            raise NotImplementedError

        if syncbn:
            norm_cfg = dict(type="SyncBN", requires_grad=True)
        else:
            norm_cfg = dict(type="BN", requires_grad=True)

        if cfg.drop_path_rate is not None:
            self.backbone = backbone(drop_path_rate=cfg.drop_path_rate, norm_cfg=norm_cfg)
        else:
            self.backbone = backbone(drop_path_rate=0.1, norm_cfg=norm_cfg)

        self.aux_head = None

        if cfg.decoder == "MLPDecoder":
            logger.info("Using MLP Decoder")
            from .decoders.MLPDecoder import DecoderHead

            self.decode_head = DecoderHead(
                in_channels=self.channels,
                num_classes=cfg.num_classes,
                norm_layer=norm_layer,
                embed_dim=cfg.decoder_embed_dim,
            )

        elif cfg.decoder == "ham":
            logger.info("Using Ham Decoder")
            print(cfg.num_classes)
            from .decoders.ham_head import LightHamHead as DecoderHead

            # from mmseg.models.decode_heads.ham_head import LightHamHead as DecoderHead
            self.decode_head = DecoderHead(
                in_channels=self.channels[1:],
                num_classes=cfg.num_classes,
                in_index=[1, 2, 3],
                norm_cfg=norm_cfg,
                channels=cfg.decoder_embed_dim,
            )
            from .decoders.fcnhead import FCNHead

            if cfg.aux_rate != 0:
                self.aux_index = 2
                self.aux_rate = cfg.aux_rate
                print("aux rate is set to", str(self.aux_rate))
                self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        else:
            logger.info("No decoder(FCN-32s)")
            from .decoders.fcnhead import FCNHead

            self.decode_head = FCNHead(
                in_channels=self.channels[-1], kernel_size=3, num_classes=cfg.num_classes, norm_layer=norm_layer
            )

    def encode_decode(self, rgb, modal_x):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgb.shape
        # print('builder',rgb.shape,modal_x.shape)
        x = self.backbone(rgb, modal_x)
        if len(x) == 2:  # if output is (rgb,depth) only use rgb
            x = x[0]
        out = self.decode_head.forward(x)
        out = F.interpolate(out, size=orisize[-2:], mode="bilinear", align_corners=False)
        if self.aux_head:
            aux_fm = self.aux_head(x[0][self.aux_index])
            aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode="bilinear", align_corners=False)
            return out, aux_fm
        return out
    
    def forward(self, rgb, modal_x=None, label=None):
        # print('builder',rgb.shape,modal_x.shape)
        if self.aux_head:
            out, aux_fm = self.encode_decode(rgb, modal_x)
        else:
            out = self.encode_decode(rgb, modal_x)

        return out
    

class DummyCfg:
    def __init__(self):
        self.backbone = "DFormer-Tiny"
        self.decoder = "ham"  # hoặc "MLPDecoder"
        self.num_classes = 21
        self.decoder_embed_dim = 128
        self.aux_rate = 0.4  # nếu dùng aux head
        self.drop_path_rate = 0.1
        self.background = 0  # class background

cfg = DummyCfg()