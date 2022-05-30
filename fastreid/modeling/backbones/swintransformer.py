# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .swin_mlp import SwinMLP
import torch
from fastreid.modeling.backbones import BACKBONE_REGISTRY
@BACKBONE_REGISTRY.register()
def build_SwinTransformer_backbone(cfg):
    model_type = cfg.MODEL.TYPE
    if model_type == 'swin':
        model = SwinTransformer(img_size=cfg.INPUT.SIZE_TRAIN,
                                patch_size=cfg.MODEL.SWIN.PATCH_SIZE,
                                in_chans=cfg.MODEL.SWIN.IN_CHANS,
                                num_classes=cfg.MODEL.NUM_CLASSES,
                                embed_dim=cfg.MODEL.SWIN.EMBED_DIM,
                                depths=cfg.MODEL.SWIN.DEPTHS,
                                num_heads=cfg.MODEL.SWIN.NUM_HEADS,
                                window_size=cfg.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=cfg.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=cfg.MODEL.SWIN.QKV_BIAS,
                                qk_scale=cfg.MODEL.SWIN.QK_SCALE,
                                drop_rate=cfg.MODEL.DROP_RATE,
                                drop_path_rate=cfg.MODEL.DROP_PATH_RATE,
                                ape=cfg.MODEL.SWIN.APE,
                                patch_norm=cfg.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=cfg.TRAIN.USE_CHECKPOINT)
    elif model_type == 'swin_mlp':
        model = SwinMLP(img_size=cfg.INPUT.SIZE_TRAIN,
                        patch_size=cfg.MODEL.SWIN_MLP.PATCH_SIZE,
                        in_chans=cfg.MODEL.SWIN_MLP.IN_CHANS,
                        num_classes=cfg.MODEL.NUM_CLASSES,
                        embed_dim=cfg.MODEL.SWIN_MLP.EMBED_DIM,
                        depths=cfg.MODEL.SWIN_MLP.DEPTHS,
                        num_heads=cfg.MODEL.SWIN_MLP.NUM_HEADS,
                        window_size=cfg.MODEL.SWIN_MLP.WINDOW_SIZE,
                        mlp_ratio=cfg.MODEL.SWIN_MLP.MLP_RATIO,
                        drop_rate=cfg.MODEL.DROP_RATE,
                        drop_path_rate=cfg.MODEL.DROP_PATH_RATE,
                        ape=cfg.MODEL.SWIN_MLP.APE,
                        patch_norm=cfg.MODEL.SWIN_MLP.PATCH_NORM,
                        use_checkpoint=cfg.TRAIN.USE_CHECKPOINT)
    
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")
    # print(torch.load("/home/lzy/.cache/torch/checkpoints/swin_base_patch4_window7_224.pth").keys())
    miss = model.load_state_dict(torch.load("/home/lzy/.cache/torch/checkpoints/swin_base_patch4_window7_224.pth")['model'],strict=False)
    print(miss)
    return model