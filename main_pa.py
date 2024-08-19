import argparse
import torch
# dassl 相关的模块，用于设置日志、随机种子、环境信息收集和构建训练器。
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import ensemble.PA.datasets.oxford_pets
import ensemble.PA.datasets.oxford_flowers
import ensemble.PA.datasets.fgvc_aircraft
import ensemble.PA.datasets.dtd
import ensemble.PA.datasets.eurosat
import ensemble.PA.datasets.stanford_cars
import ensemble.PA.datasets.food101
import ensemble.PA.datasets.sun397
import ensemble.PA.datasets.caltech101
import ensemble.PA.datasets.ucf101
import ensemble.PA.datasets.imagenet

import ensemble.PA.datasets.imagenet_sketch
import ensemble.PA.datasets.imagenetv2
import ensemble.PA.datasets.imagenet_a
import ensemble.PA.datasets.imagenet_r
import ensemble.PA.datasets.pug

# import trainers.maple
import ensemble.PA.trainers.prompt_align

from pdb import set_trace as stx

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN
    
    # Config for MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.TPT = 'I'

    # TPT args
    cfg.TPT = CN()
    cfg.TPT.LOADER = True   # Use TPT Dataloader. (Just for sanity check)
    cfg.TPT.RUN = True  # Run TPT using TPT dataloader
    cfg.TPT.LR = 4e-2   # Learning rate for TPT
    cfg.TPT.COCOOP = False
    cfg.TPT.ALIGN_LAYER_FROM = 0
    cfg.TPT.ALIGN_LAYER_TO = 3
    cfg.TPT.TTA_STEPS = 1
    cfg.TPT.DISTR_ALIGN = False
    cfg.TPT.TPT_THRESHOLD = 0.1
    cfg.TPT.ALIGN_THRESHOLD = 0.1
    cfg.TPT.TPT_LOSS = True
    cfg.TPT.DISTR_LOSS_W = 100.0
    cfg.TPT.BATCH_SIZE = 64
    cfg.TPT.VIS_MEANS = './output/PromptAlign/features/ImgNet_vis_means.pt'  # Path to means of source dataset for vision branch
    cfg.TPT.VIS_VARS = './output/PromptAlign/features/ImgNet_vis_vars.pt'    # Path to variances of source dataset for vision branch

    # Config for MaPLe
    cfg.TRAINER.PROMPTALIGN = CN()
    cfg.TRAINER.PROMPTALIGN.N_CTX = 2  # number of context vectors
    cfg.TRAINER.PROMPTALIGN.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.PROMPTALIGN.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PROMPTALIGN.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.TPT = 'I'
    cfg.DATASET.VARIANT = 'Worlds'  # Added for PUG dataset variants


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # 打印参数和配置
    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))
    # TPT 标志检查：如果命令行参数中设置了 tpt，检查它是否与配置中的 TPT.RUN 一致，确保没有设置冲突。
    if args.tpt:
        assert args.tpt == cfg.TPT.RUN, "TPT flag in args and config mismatch"
    # 根据配置使用 build_trainer(cfg) 函数构建训练器。
    #print("args.VIS_VARS",args.VIS_VARS)
    trainer = build_trainer(cfg)
    # 如果设置了 args.eval_only，则加载模型并进行评估，不执行训练过程。
    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return
    elif args.tpt: # 果设置了 tpt 参数，加载模型并执行测试时间提示调整（Test Time Prompt Tuning，TPT）。trainer.tpt() 执行 TPT 并返回结果。
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        results = trainer.tpt()   # Perform TPT and inference
        print()
        print("\t\t [set_id] \t\t Top-1 acc. \t\t Top-5 acc.")
        for id in results.keys():
            print("{}".format(id), end="	")
        print("\n")
        for id in results.keys():
            print("{:.2f}".format(results[id][0]), end="	")
        print("\n")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument('--tpt', action='store_true', default=True, help='run test-time prompt tuning')
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
