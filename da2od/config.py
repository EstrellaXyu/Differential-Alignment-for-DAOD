from detectron2.config import CfgNode as CN


def add_da2od_config(cfg):
    _C = cfg

    # Datasets and sampling
    _C.DATASETS.UNLABELED = tuple()
    _C.DATASETS.BATCH_COMPONENTS = ("labeled_weak", ) # one or more of: { "labeled_weak", "labeled_strong", "unlabeled_weak", "unlabeled_strong" }
    _C.DATASETS.BATCH_RATIOS = (1,) # must match length of BATCH_COMPONENTS
    # change here
    _C.DATASETS.BLEND = False
    # end change

    # Strong augmentations
    _C.AUG = CN()
    _C.AUG.WEAK_INCLUDES_MULTISCALE = True
    _C.AUG.LABELED_INCLUDE_RANDOM_ERASING = True
    _C.AUG.UNLABELED_INCLUDE_RANDOM_ERASING = True
    _C.AUG.LABELED_MIC_AUG = False
    _C.AUG.UNLABELED_MIC_AUG = False
    _C.AUG.MIC_RATIO = 0.5
    _C.AUG.MIC_BLOCK_SIZE = 32

    # EMA of student weights
    _C.EMA = CN()
    _C.EMA.ENABLED = False
    _C.EMA.ALPHA = 0.9996
    _C.EMA.LOAD_FROM_EMA_ON_START = True

    # Begin domain adaptation settings
    _C.DA = CN()

    # Source-target alignment
    _C.DA.ALIGN = CN()
    _C.DA.ALIGN.IMG_DA_ENABLED = False
    _C.DA.ALIGN.IMG_DA_LAYER = "p2"
    _C.DA.ALIGN.IMG_DA_WEIGHT = 0.01
    _C.DA.ALIGN.IMG_DA_INPUT_DIM = 256 # = output channels of backbone
    _C.DA.ALIGN.IMG_DA_HIDDEN_DIMS = [256,]
    _C.DA.ALIGN.INS_DA_ENABLED = False
    _C.DA.ALIGN.INS_DA_WEIGHT = 0.01
    _C.DA.ALIGN.INS_DA_INPUT_DIM = 1024 # = output channels of box head
    _C.DA.ALIGN.INS_DA_HIDDEN_DIMS = [1024,]

    # Self-distillation
    _C.DA.DISTILL = CN()
    _C.DA.DISTILL.DISTILLER_NAME = "DA2ODDistiller"
    # 'Pseudo label' approaches
    _C.DA.DISTILL.HARD_ROIH_CLS_ENABLED = False
    _C.DA.DISTILL.HARD_ROIH_REG_ENABLED = False
    _C.DA.DISTILL.HARD_OBJ_ENABLED = False
    _C.DA.DISTILL.HARD_RPN_REG_ENABLED = False
    # 'Distillation' approaches
    _C.DA.DISTILL.ROIH_CLS_ENABLED = False
    _C.DA.DISTILL.ROIH_REG_ENABLED = False
    _C.DA.DISTILL.OBJ_ENABLED = False
    _C.DA.DISTILL.RPN_REG_ENABLED = False
    _C.DA.DISTILL.CLS_TMP = 1.0
    _C.DA.DISTILL.OBJ_TMP = 1.0
    _C.DA.CLS_LOSS_TYPE = "CE" # one of: { "CE", "KL" }

    _C.DA.TEACHER = CN()
    _C.DA.TEACHER.ENABLED = False
    _C.DA.TEACHER.THRESHOLD = 0.8

    # num_gradient_accum_steps = IMS_PER_BATCH / (NUM_GPUS * IMS_PER_GPU)
    _C.SOLVER.IMS_PER_GPU = 2

    _C.SOLVER.BACKWARD_AT_END = True

    # Enable use of different optimizers (necessary to match VitDet settings)
    _C.SOLVER.OPTIMIZER = "SGD"