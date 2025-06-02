from datetime import timedelta

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results

from da2od.ema import EMA
from da2od.checkpoint import CheckpointerWithEMA
from da2od.config import add_da2od_config
from da2od.trainer import DA2ODTrainer
import da2od.datasets # register datasets with Detectron2
import da2od.model # register da2od R-CNN model with Detectron2


def setup(args):
    """
    Copied directly from detectron2/tools/train_net.py
    """
    cfg = get_cfg()

    add_da2od_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    """
    Copied directly from detectron2/tools/train_net.py
    But replace Trainer with DATrainer and disable TTA.
    """
    cfg = setup(args)

    if args.eval_only:
        model = DA2ODTrainer.build_model(cfg)
        ## Change here
        ckpt = CheckpointerWithEMA(model, save_dir=cfg.OUTPUT_DIR)
        if cfg.EMA.ENABLED and cfg.EMA.LOAD_FROM_EMA_ON_START:
            ema = EMA(DA2ODTrainer.build_model(cfg), cfg.EMA.ALPHA)
            ckpt.add_checkpointable("ema", ema)
        ckpt.resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        ## End change
        res = DA2ODTrainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            raise NotImplementedError("TTA not supported")
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = DA2ODTrainer(cfg)
    trainer.resume_or_load(resume=True)
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        timeout=timedelta(minutes=1),
        args=(args,),
    )