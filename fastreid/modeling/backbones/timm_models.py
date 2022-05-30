import re
import timm
from fastreid.layers import get_norm
from .build import BACKBONE_REGISTRY
from .mobilenet import _make_divisible

@BACKBONE_REGISTRY.register()
def build_timm_backbone(cfg):
    model_type = cfg.MODEL.NAME #'seresnet152d'
    model = timm.create_model(model_type, pretrained=True)
    return model