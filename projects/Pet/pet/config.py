from fastreid.config import CfgNode as CN
def add_pet_config(cfg):
    _C = cfg
    _C.MODEL.TRANSFORMER = CN()
    _C.MODEL.TRANSFORMER.DIM_MODEL = 512
    _C.MODEL.TRANSFORMER.ENCODER_LAYERS = 1
    _C.MODEL.TRANSFORMER.N_HEAD = 8
    _C.MODEL.TRANSFORMER.USE_OUTPUT_LAYER = False
    _C.MODEL.TRANSFORMER.DROPOUT = 0.
    _C.MODEL.TRANSFORMER.USE_LOCAL_SHORTCUT = True
    _C.MODEL.TRANSFORMER.USE_GLOBAL_SHORTCUT = True

    _C.MODEL.TRANSFORMER.USE_DIFF_SCALE = True
    _C.MODEL.TRANSFORMER.TRANS_NAMES = ['scale1','scale2']
    _C.MODEL.TRANSFORMER.NAMES_1ST = ['scale1','scale2']
    _C.MODEL.TRANSFORMER.NAMES_2ND = ['scale1','scale2']
    _C.MODEL.TRANSFORMER.NAMES_3RD = ['scale1','scale2']
    _C.MODEL.TRANSFORMER.KERNEL_SIZE_1ST = [(1,1),(3,3)]
    _C.MODEL.TRANSFORMER.KERNEL_SIZE_2ND = [(1,1),(3,3)]
    _C.MODEL.TRANSFORMER.KERNEL_SIZE_3RD = [(1,1),(3,3)]
    _C.MODEL.TRANSFORMER.USE_MASK_1ST = False
    _C.MODEL.TRANSFORMER.USE_MASK_2ND = True
    _C.MODEL.TRANSFORMER.USE_MASK_3RD = True
    _C.MODEL.TRANSFORMER.USE_PATCH2VEC = True

    ####
    _C.MODEL.USE_FEATURE_MASK = True
    _C.MODEL.FEATURE_AUG_TYPE = 'exchange_token' # 'exchange_token', 'jigsaw_token', 'cutout_patch', 'erase_patch', 'mixup_patch', 'jigsaw_patch'
    _C.MODEL.FEATURE_MASK_SIZE = 4
    _C.MODEL.MASK_SHAPE = 'stripe' # 'square', 'random'
    _C.MODEL.MASK_SIZE = 1
    _C.MODEL.MASK_MODE = 'random_direction' # 'horizontal', 'vertical' for stripe; 'random_size' for square
    _C.MODEL.MASK_PERCENT = 0.1
    #### 
    _C.MODEL.EMBEDDING_DIM = 256
