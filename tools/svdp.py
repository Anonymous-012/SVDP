import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction
from copy import deepcopy

from mmseg.apis import multi_gpu_test, single_gpu_test, single_gpu_svdp
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from IPython import embed
import random
import numpy as np
import wandb
def set_random_seed(seed=1, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_ema_model(model):
    ema_model = deepcopy(model) # get_model(args.model)(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    return ema_model




def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('--config', default = "local_configs/segformer/B5/segformer.b5.1024x1024.acdc.160k_p.py", help='test config file path')
    parser.add_argument('--checkpoint', default ="segformer.b5.1024x1024.city.160k.pth", help='checkpoint file')

    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', default='work_dirs/res.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default='mIoU',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--scale', type=float, default=0.1)
    parser.add_argument('--model_lr', type=float, default=3e-4)
    parser.add_argument('--prompt_lr', type=float, default=1e-4)
    parser.add_argument('--prompt_sparse_rate', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ema_rate', type=float, default=0.999)
    parser.add_argument('--ema_rate_prompt', type=float, default=0.999)
    parser.add_argument('--wandb_login', type=str)
    parser.add_argument('--wandb_project', type=str)
    parser.add_argument('--wandb_name', type=str, default="debug")

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    set_random_seed(seed=args.seed)
    wandb.init(
        name=args.wandb_name,
        project=args.wandb_project,
        entity=args.wandb_login,
        mode="online",
        save_code=True,
        config=args,
    )
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>init param>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print("model_lr", args.model_lr)
    print("prompt_lr", args.prompt_lr)
    print("prompt_sparse_rate", args.prompt_sparse_rate)
    print("ema_rate", args.ema_rate)
    print("seed", args.seed)
    print("scale", args.scale)

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>over param>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if 'None' in args.eval:
        args.eval = None
    if args.eval and args.format_only:

        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if True:#True: #args.aug_test:
        if cfg.data.test.type in ['CityscapesDataset', 'ACDCDataset']:
            # hard code index
            cfg.data.test.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
            ]
            cfg.data.test.pipeline[1].flip = True
        elif cfg.data.test.type == 'ADE20KDataset':
            # hard code index
            cfg.data.test.pipeline[1].img_ratios = [
                0.75, 0.875, 1.0, 1.125, 1.25
            ]
            cfg.data.test.pipeline[1].flip = True
        else:
            # hard code index
            cfg.data.test.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75
            ]
            cfg.data.test.pipeline[1].flip = True
        if cfg.data.test1.type in ['CityscapesDataset', 'ACDCDataset']:
            # hard code index
            cfg.data.test1.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
            ]
            cfg.data.test1.pipeline[1].flip = True
        elif cfg.data.test1.type == 'ADE20KDataset':
            # hard code index
            cfg.data.test1.pipeline[1].img_ratios = [
                0.75, 0.875, 1.0, 1.125, 1.25
            ]
            cfg.data.test1.pipeline[1].flip = True
        else:
            # hard code index
            cfg.data.test1.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75
            ]
            cfg.data.test1.pipeline[1].flip = True
        if cfg.data.test2.type in ['CityscapesDataset', 'ACDCDataset']:
            # hard code index
            cfg.data.test2.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
            ]
            cfg.data.test2.pipeline[1].flip = True
        elif cfg.data.test2.type == 'ADE20KDataset':
            # hard code index
            cfg.data.test2.pipeline[1].img_ratios = [
                0.75, 0.875, 1.0, 1.125, 1.25
            ]
            cfg.data.test2.pipeline[1].flip = True
        else:
            # hard code index
            cfg.data.test2.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75
            ]
            cfg.data.test2.pipeline[1].flip = True
        if cfg.data.test3.type in ['CityscapesDataset', 'ACDCDataset']:
            # hard code index
            cfg.data.test3.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
            ]
            cfg.data.test3.pipeline[1].flip = True
        elif cfg.data.test.type == 'ADE20KDataset':
            # hard code index
            cfg.data.test3.pipeline[1].img_ratios = [
                0.75, 0.875, 1.0, 1.125, 1.25
            ]
            cfg.data.test3.pipeline[1].flip = True
        else:
            # hard code index
            cfg.data.test3.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75
            ]
            cfg.data.test3.pipeline[1].flip = True


    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    print(cfg)
    datasets = [build_dataset(cfg.data.test), build_dataset(cfg.data.test1), build_dataset(cfg.data.test2), build_dataset(cfg.data.test3)]
    cfg.model.train_cfg = None

    cfg.model.backbone.prompt_sparse_rate = args.prompt_sparse_rate

    
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']

    model = MMDataParallel(model, device_ids=[0])
    anchor = deepcopy(model.state_dict())
    anchor_model = deepcopy(model)
    ema_model = create_ema_model(model)

    efficient_test = True #False
    # if args.eval_options is not None:
    #     efficient_test = args.eval_options.get('efficient_test', False)


    # tuning on continual step
    cnt = 0
    num_itr = 3

    All_mIoU = 0
    for i in range(num_itr):
        mean_mIoU = 0
        print("revisiting", i)
        data_loaders = [build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False) for dataset in datasets]

        for dataset, data_loader in zip(datasets, data_loaders):
            outputs = single_gpu_svdp(args, model, data_loader, args.show, args.show_dir,
                                  efficient_test, anchor, ema_model, anchor_model, False, False)
            rank, _ = get_dist_info()
            if rank == 0:
                if args.out:
                    print(f'\nwriting results to {args.out}')
                    mmcv.dump(outputs, args.out)
                kwargs = {} if args.eval_options is None else args.eval_options
                if args.format_only:
                    dataset.format_results(outputs, **kwargs)
                if args.eval:
                    results = dataset.evaluate(outputs, args.eval, **kwargs)
                    mIoU = results['mIoU']
                    wandb.log(
                        {
                            "mIoU": mIoU,
                        }
                    )
                    print('1')
                    mean_mIoU += mIoU

        wandb.log(
            {
                "mean_mIoU": mean_mIoU/4,
            }
        )
        All_mIoU = All_mIoU + mean_mIoU/4
    wandb.log(
        {
            "All_mIoU": All_mIoU/num_itr,
        }
    )
    wandb.finish()

if __name__ == '__main__':
    main()
