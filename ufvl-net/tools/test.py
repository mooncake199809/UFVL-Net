# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
from numbers import Number

import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model, CheckpointLoader, load_state_dict)

from ufvl_net.apis import multi_gpu_test, single_gpu_test
from ufvl_net.datasets import build_dataloader, build_dataset
from ufvl_net.models import build_architecture
from ufvl_net.utils import get_root_logger, setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(description='ufvl_net test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    out_options = ['class_scores', 'pred_score', 'pred_label', 'pred_class']
    parser.add_argument(
        '--out-items',
        nargs='+',
        default=['all'],
        choices=out_options + ['none', 'all'],
        help='Besides metrics, what items will be included in the output '
        f'result file. You can choose some of ({", ".join(out_options)}), '
        'or use "all" to include all above, or use "none" to disable all of '
        'above. Defaults to output all.',
        metavar='')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., '
        '"accuracy", "precision", "recall", "f1_score", "support" for single '
        'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
        'multi-label dataset')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--metric-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be parsed as a dict metric_options for dataset.evaluate()'
        ' function.')
    parser.add_argument(
        '--show-options',
        nargs='+',
        action=DictAction,
        help='custom options for show_result. key-value pair in xxx=yyy.'
        'Check available options in `model.show_result`.')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda', 'ipu'],
        default='cuda',
        help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    assert args.metrics or args.out, \
        'Please specify at least one of output path and evaluation metrics.'

    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    dataset = build_dataset(cfg.data.test, default_args=dict(test_mode=True))

    # build the dataloader
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=1 if args.device == 'ipu' else len(cfg.gpu_ids),
        dist=distributed,
        round_up=True,
    )
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'shuffle': False,  # Not shuffle by default
        'sampler_cfg': None,  # Not use sampler by default
        **cfg.data.get('test_dataloader', {}),
    }
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    model = build_architecture(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    def handle_checkpoint(model_path, 
                          map_location=None,
                          strict=False,
                          logger=None,
                          revise_keys=[(r'^module\.', '')]):
        from collections import OrderedDict
        import re
        threshold = 0.5

        all_scene = []
        for scene in cfg.all_scene:
            if cfg.scene == scene:
                continue
            else:
                all_scene.append(scene)

        assert cfg.share_type == "channel" or cfg.share_type == "kernel", print("Sharing Type Error")
        new_checkpoint = {}
        checkpoint = CheckpointLoader.load_checkpoint(model_path, map_location="cpu")
        for checkpoint_name in checkpoint.keys():
            if cfg.scene in checkpoint_name:
                new_checkpoint.update({checkpoint_name[checkpoint_name.find("_")+1:] : checkpoint[checkpoint_name]})
            elif any(scene in checkpoint_name for scene in all_scene):
                continue
            else:
                new_checkpoint.update({checkpoint_name : checkpoint[checkpoint_name]})
        checkpoint = new_checkpoint
        new_checkpoint = {}
        specific_name_list = ["downsample.1", "head", "se_layer", "bn"]
        for checkpoint_name in checkpoint.keys():
            if "score" in checkpoint_name:
                if cfg.share_type == "channel":
                    shared_index = torch.where(checkpoint[checkpoint_name] <= threshold)[0]
                    specif_index = torch.where(checkpoint[checkpoint_name] > threshold)[0]
                    new_tensor = torch.ones_like(torch.cat((checkpoint[checkpoint_name[:checkpoint_name.rfind(".")] + ".weight"],
                                                            checkpoint[checkpoint_name[:checkpoint_name.rfind(".")] + ".specific_weight"]),
                                                            dim=1))
                    new_tensor[:,shared_index,:,:] = checkpoint[checkpoint_name[:checkpoint_name.rfind(".")] + ".weight"]
                    new_tensor[:,specif_index,:,:] = checkpoint[checkpoint_name[:checkpoint_name.rfind(".")] + ".specific_weight"]
                    new_checkpoint.update({
                        checkpoint_name[:checkpoint_name.rfind(".")] + ".weight" : new_tensor
                    })
                elif cfg.share_type == "kernel":
                    shared_index = torch.where(checkpoint[checkpoint_name] <= threshold)
                    specif_index = torch.where(checkpoint[checkpoint_name] > threshold)
                    Cout, Cin, N1 = checkpoint[checkpoint_name[:checkpoint_name.rfind(".")] + ".weight"].shape
                    N2 = checkpoint[checkpoint_name[:checkpoint_name.rfind(".")] + ".specific_weight"].shape[-1]
                    kernel_size = int((N1 + N2) ** 0.5)
                    new_tensor = torch.ones(Cout, Cin, kernel_size, kernel_size).type_as(checkpoint[checkpoint_name[:checkpoint_name.rfind(".")] + ".weight"])
                    new_tensor[:,:,shared_index[0],shared_index[1]] = checkpoint[checkpoint_name[:checkpoint_name.rfind(".")] + ".weight"]
                    new_tensor[:,:,specif_index[0],specif_index[1]] = checkpoint[checkpoint_name[:checkpoint_name.rfind(".")] + ".specific_weight"]
                    new_checkpoint.update({
                        checkpoint_name[:checkpoint_name.rfind(".")] + ".weight" : new_tensor
                    })
                else:
                    print("Sharing TYPE ERROR")
                    exit()
            elif any(specific_name in checkpoint_name for specific_name in specific_name_list):
                new_checkpoint.update({
                    checkpoint_name : checkpoint[checkpoint_name]
                })
        checkpoint = new_checkpoint

        if not isinstance(checkpoint, dict):
            raise RuntimeError(
                f'No state_dict found in checkpoint file {model_path}')
        # get state_dict from checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # strip prefix of state_dict
        metadata = getattr(state_dict, '_metadata', OrderedDict())
        for p, r in revise_keys:
            state_dict = OrderedDict(
                {re.sub(p, r, k): v
                for k, v in state_dict.items()})
        # Keep metadata in state_dict
        state_dict._metadata = metadata

        # load state_dict
        load_state_dict(model, state_dict, strict, logger)
        return new_checkpoint

    checkpoint = handle_checkpoint(args.checkpoint)

    if 'CLASSES' in checkpoint.get('meta', {}):
        CLASSES = checkpoint['meta']['CLASSES']

    if not distributed:
        if args.device == 'cpu':
            model = model.cpu()
        elif args.device == 'ipu':
            from mmcv.device.ipu import cfg2options, ipu_model_wrapper
            opts = cfg2options(cfg.runner.get('options_cfg', {}))
            if fp16_cfg is not None:
                model.half()
            model = ipu_model_wrapper(model, opts, fp16_cfg=fp16_cfg)
            data_loader.init(opts['inference'])
        else:
            model = MMDataParallel(model, device_ids=cfg.gpu_ids)
            if not model.device_ids:
                assert mmcv.digit_version(mmcv.__version__) >= (1, 4, 4), \
                    'To test with CPU, please confirm your mmcv version ' \
                    'is not lower than v1.4.4'
        show_kwargs = {} if args.show_options is None else args.show_options
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  **show_kwargs)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        results = {}
        logger = get_root_logger()
        if args.metrics:
            eval_results = dataset.evaluate(
                results=outputs,
                metric=args.metrics,
                metric_options=args.metric_options,
                logger=logger)
            results.update(eval_results)
            for k, v in eval_results.items():
                if isinstance(v, np.ndarray):
                    v = [round(out, 2) for out in v.tolist()]
                elif isinstance(v, Number):
                    v = round(v, 2)
                else:
                    raise ValueError(f'Unsupport metric type: {type(v)}')
                print(f'\n{k} : {v}')
        if args.out:
            if 'none' not in args.out_items:
                scores = np.vstack(outputs)
                pred_score = np.max(scores, axis=1)
                pred_label = np.argmax(scores, axis=1)
                pred_class = [CLASSES[lb] for lb in pred_label]
                res_items = {
                    'class_scores': scores,
                    'pred_score': pred_score,
                    'pred_label': pred_label,
                    'pred_class': pred_class
                }
                if 'all' in args.out_items:
                    results.update(res_items)
                else:
                    for key in args.out_items:
                        results[key] = res_items[key]
            print(f'\ndumping results to {args.out}')
            mmcv.dump(results, args.out)


if __name__ == '__main__':
    main()
