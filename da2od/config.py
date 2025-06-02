from detectron2.config import CfgNode

def add_da2od_config(cfg):
    _C = cfg

    # Datasets and sampling
    _C.DATASETS.UNLABELED = tuple()
    _C.DATASETS.LABELED = tuple()
    _C.DATASETS.BATCH_COMPONENTS = ("labeled_weak", )
    _C.DATASETS.BATCH_RATIOS = (1,)        

    # Strong augs
    _C.AUG = CfgNode()
    _C.AUG.WEAK_INCLUDES_MULTISCALE = True
    _C.AUG.LABELED_INCLUDE_RANDOM_ERASING = True
    _C.AUG.UNLABELED_INCLUDE_RANDOM_ERASING = True
    _C.AUG.LABELED_MIC_AUG = False
    _C.AUG.UNLABELED_MIC_AUG = False
    _C.AUG.MIC_RATIO = 0.5
    _C.AUG.MIC_BLOCK_SIZE = 32

    # EMA
    _C.EMA = CfgNode()
    _C.EMA.ENABLED = False
    _C.EMA.ALPHA = 0.9996
    _C.EMA.LOAD_FROM_EMA_ON_START = True

    # DA settingss
    _C.DA = CfgNode()
    # alignment settings
    _C.DA.ALIGN = CfgNode()
    _C.DA.ALIGN.IMG_ALIGN_ENABLED = False
    _C.DA.ALIGN.IMG_ALIGN_LAYER = "p2"
    _C.DA.ALIGN.IMG_ALIGN_WEIGHT = 0.01
    _C.DA.ALIGN.IMG_ALIGN_INPUT_DIM = 256 # = output channels of backbone
    _C.DA.ALIGN.IMG_ALIGN_HIDDEN_DIMS = [256,]
    _C.DA.ALIGN.INS_ALIGN_ENABLED = False
    _C.DA.ALIGN.INS_ALIGN_WEIGHT = 0.01
    _C.DA.ALIGN.INS_ALIGN_INPUT_DIM = 1024 # = output channels of box head
    _C.DA.ALIGN.INS_ALIGN_HIDDEN_DIMS = [1024,]

    # distillation settings
    _C.DA.DISTILL = CfgNode()
    _C.DA.DISTILL.DISTILLER_NAME = "DA2ODDistiller"
    # hard distill
    _C.DA.DISTILL.HARD_ROIH_CLS_ENABLED = False
    _C.DA.DISTILL.HARD_ROIH_REG_ENABLED = False
    _C.DA.DISTILL.HARD_OBJ_ENABLED = False
    _C.DA.DISTILL.HARD_RPN_REG_ENABLED = False
    # soft distill
    _C.DA.DISTILL.SOFT_ROIH_CLS_ENABLED = False
    _C.DA.DISTILL.SOFT_ROIH_REG_ENABLED = False
    _C.DA.DISTILL.SOFT_OBJ_ENABLED = False
    _C.DA.DISTILL.SOFT_RPN_REG_ENABLED = False
    _C.DA.DISTILL.CLS_TMP = 1.0
    _C.DA.DISTILL.OBJ_TMP = 1.0
    _C.DA.CLS_LOSS_TYPE = "CE"

    # Teacher model provides pseudo labels
    _C.DA.TEACHER = CfgNode()
    _C.DA.TEACHER.ENABLED = False
    _C.DA.TEACHER.THRESHOLD = 0.8

    # num_gradient_accum_steps = IMS_PER_BATCH / (NUM_GPUS * IMS_PER_GPU)
    _C.SOLVER.IMS_PER_GPU = 2

    # every phrase(including forward, distill, align) backward, aims to save memory
    _C.SOLVER.BACKWARD_EVERY_PHASE= False

    _C.SOLVER.OPTIMIZER = "SGD"