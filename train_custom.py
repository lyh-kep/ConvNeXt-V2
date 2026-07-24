import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
import datetime
import numpy as np
import time
import json
import csv
from pathlib import Path
import sys

import torch
import torch.backends.cudnn as cudnn
import torchvision

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma

from datasets import build_dataset
from engine_finetune import train_one_epoch, evaluate
from optim_factory import create_optimizer, LayerDecayValueAssigner

import utils
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import str2bool, remap_checkpoint_keys
import models.convnextv2 as convnextv2

def get_args_parser():
    parser = argparse.ArgumentParser('FCMAE fine-tuning', add_help=False)
    
    parser.add_argument('--data_path', default='dataset/', type=str,
                        help='训练数据集路径，目录下包含类别子目录')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='验证数据集路径，默认使用data_path下的val目录')
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='类别数量，二分类设为2')
    parser.add_argument('--data_set', default='image_folder', choices=['CIFAR', 'IMNET', 'image_folder'], type=str,
                        help='数据集类型，image_folder表示按文件夹组织的图片')
    
    parser.add_argument('--model', default='convnextv2_atto', type=str,
                        help='模型名称，支持: convnextv2_atto, convnextv2_femto, convnextv2_nano, '
                             'convnextv2_tiny, convnextv2_base, convnextv2_large, convnextv2_huge')
    parser.add_argument('--input_size', default=224, type=int,
                        help='输入图像尺寸，默认224')
    parser.add_argument('--finetune', default='weights/convnextv2_atto_1k_224_fcmae.pt', type=str,
                        help='预训练权重路径')
    
    parser.add_argument('--batch_size', default=8, type=int,
                        help='批大小')
    parser.add_argument('--epochs', default=2, type=int,
                        help='训练轮数')
    parser.add_argument('--blr', type=float, default=2e-4,
                        help='基础学习率 (batch_size=256时的学习率)')
    parser.add_argument('--layer_decay', type=float, default=0.9,
                        help='分层学习率衰减系数，越小表示低层学习率越低')
    
    parser.add_argument('--output_dir', default='runs/train/', type=str,
                        help='输出目录，自动创建exp/exp2/exp3等子目录')
    parser.add_argument('--device', default='cuda', type=str,
                        help='训练设备，cuda或cpu')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='数据加载线程数，Windows建议设为0')
    parser.add_argument('--use_amp', type=str2bool, default=False,
                        help='是否使用混合精度训练')
    
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='随机深度概率')
    parser.add_argument('--layer_decay_type', type=str, choices=['single', 'group'], default='single')
    
    parser.add_argument('--model_ema', type=str2bool, default=True,
                        help='是否使用EMA模型')
    parser.add_argument('--model_ema_decay', type=float, default=0.9999)
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False)
    parser.add_argument('--model_ema_eval', type=str2bool, default=True)

    parser.add_argument('--clip_grad', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    
    parser.add_argument('--warmup_steps', type=int, default=-1)
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--opt_eps', default=1e-8, type=float)
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay_end', type=float, default=None)

    parser.add_argument('--color_jitter', type=float, default=None)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1')
    parser.add_argument('--smoothing', type=float, default=0.2)
    
    parser.add_argument('--train_interpolation', type=str, default='bicubic')

    parser.add_argument('--reprob', type=float, default=0.25)
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--resplit', type=str2bool, default=False)

    parser.add_argument('--mixup', type=float, default=0.0)
    parser.add_argument('--cutmix', type=float, default=0.0)
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None)
    parser.add_argument('--mixup_prob', type=float, default=1.0)
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5)
    parser.add_argument('--mixup_mode', type=str, default='batch')

    parser.add_argument('--head_init_scale', default=0.001, type=float)
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)

    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', type=str)
    
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--auto_resume', type=str2bool, default=True)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)

    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval', type=str2bool, default=False)
    parser.add_argument('--dist_eval', type=str2bool, default=False)
    parser.add_argument('--disable_eval', type=str2bool, default=False)
    parser.add_argument('--pin_mem', type=str2bool, default=True)

    parser.add_argument('--crop_pct', type=float, default=None)

    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://', type=str)
    
    return parser

def get_next_exp_dir(base_dir='train'):
    os.makedirs(base_dir, exist_ok=True)
    existing_dirs = [d for d in os.listdir(base_dir) if d.startswith('exp')]
    if not existing_dirs:
        return os.path.join(base_dir, 'exp')
    max_num = 1
    for d in existing_dirs:
        try:
            num = int(d[3:]) if len(d) > 3 else 1
            max_num = max(max_num, num + 1)
        except:
            pass
    if max_num == 1:
        return os.path.join(base_dir, 'exp')
    else:
        return os.path.join(base_dir, f'exp{max_num}')

def plot_results(output_dir, history, num_classes=2):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import roc_curve, auc
        sns.set_style('darkgrid')
        epochs = [h['epoch'] for h in history]
        
        if num_classes == 2:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            axes[0, 0].plot(epochs, [h.get('train_loss', 0) for h in history], label='Train Loss', color='blue')
            axes[0, 0].plot(epochs, [h.get('test_loss', 0) for h in history], label='Val Loss', color='red')
            axes[0, 0].set_title('Loss')
            axes[0, 0].legend()
            
            axes[0, 1].plot(epochs, [h.get('train_class_acc', 0) for h in history], label='Train Acc', color='blue')
            axes[0, 1].plot(epochs, [h.get('test_acc1', 0) for h in history], label='Val Acc', color='red')
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].legend()
            
            axes[1, 0].plot(epochs, [h.get('train_lr', 0) for h in history], label='Learning Rate', color='green')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].legend()
            
            if 'test_acc1_ema' in history[0]:
                axes[1, 1].plot(epochs, [h.get('test_acc1', 0) for h in history], label='Model Acc', color='blue')
                axes[1, 1].plot(epochs, [h.get('test_acc1_ema', 0) for h in history], label='EMA Acc', color='orange')
                axes[1, 1].set_title('Model vs EMA Accuracy')
                axes[1, 1].legend()
            else:
                axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'results.png'), dpi=150)
            plt.close()
            
            print(f"Plot saved to {os.path.join(output_dir, 'results.png')}")
            
            last_history = history[-1] if history else {}
            if 'test_fpr' in last_history and 'test_tpr' in last_history:
                fpr = last_history['test_fpr']
                tpr = last_history['test_tpr']
                roc_auc = last_history.get('test_roc_auc', 0)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, lw=2, color='navy', label=f'ROC (AUC = {roc_auc:.4f})')
                plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2)
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend(loc='lower right')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150)
                plt.close()
                
                print(f"ROC curve saved to {os.path.join(output_dir, 'roc_curve.png')}")
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            axes[0, 0].plot(epochs, [h.get('train_loss', 0) for h in history], label='Train Loss', color='blue')
            axes[0, 0].plot(epochs, [h.get('test_loss', 0) for h in history], label='Val Loss', color='red')
            axes[0, 0].set_title('Loss')
            axes[0, 0].legend()
            
            axes[0, 1].plot(epochs, [h.get('train_class_acc', 0) for h in history], label='Train Acc', color='blue')
            axes[0, 1].plot(epochs, [h.get('test_acc1', 0) for h in history], label='Val Acc', color='red')
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].legend()
            
            axes[1, 0].plot(epochs, [h.get('train_lr', 0) for h in history], label='Learning Rate', color='green')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].legend()
            
            if 'test_acc1_ema' in history[0]:
                axes[1, 1].plot(epochs, [h.get('test_acc1', 0) for h in history], label='Model Acc', color='blue')
                axes[1, 1].plot(epochs, [h.get('test_acc1_ema', 0) for h in history], label='EMA Acc', color='orange')
                axes[1, 1].set_title('Model vs EMA Accuracy')
                axes[1, 1].legend()
            else:
                axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'results.png'), dpi=150)
            plt.close()
            
            print(f"Plot saved to {os.path.join(output_dir, 'results.png')}")
    except Exception as e:
        print(f"Failed to plot results: {e}")

def save_results_csv(output_dir, history):
    if not history:
        return
    keys = history[0].keys()
    csv_path = os.path.join(output_dir, 'results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    print(f"Results saved to {csv_path}")

def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    if args.disable_eval:
        args.dist_eval = False
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(is_train=False, args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    model = convnextv2.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        head_init_scale=args.head_init_scale,
    )
    
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print(f"Load pre-trained checkpoint from: {args.finetune}")
        checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                del checkpoint_model[k]
        
        checkpoint_model_keys = list(checkpoint_model.keys())
        for k in checkpoint_model_keys:
            if 'decoder' in k or 'mask_token' in k or 'proj' in k or 'pred' in k:
                del checkpoint_model[k]
        
        checkpoint_model = remap_checkpoint_keys(checkpoint_model)
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
        
        trunc_normal_(model.head.weight, std=2e-5)
        torch.nn.init.constant_(model.head.bias, 0.)

    model.to(device)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print(f"Using EMA with decay = {args.model_ema_decay:.8f}")

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model: {args.model}")
    print(f"Number of params: {n_parameters:,}")

    eff_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // eff_batch_size
    
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print(f"Base lr: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"Actual lr: {args.lr:.2e}")
    print(f"Effective batch size: {eff_batch_size}")
    print(f"Training samples: {len(dataset_train)}")
    print(f"Validation samples: {len(dataset_val) if dataset_val else 0}")

    if args.layer_decay < 1.0 or args.layer_decay > 1.0:
        assert args.layer_decay_type in ['single', 'group']
        if args.layer_decay_type == 'group':
            num_layers = 12
        else:
            num_layers = sum(model_without_ddp.depths)
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)),
            depths=model_without_ddp.depths, layer_decay_type=args.layer_decay_type)
    else:
        assigner = None

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=None,
        get_num_layer=assigner.get_layer_id if assigner is not None else None, 
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        print(f"Eval only mode")
        test_stats = evaluate(data_loader_val, model, device, num_classes=args.nb_classes)
        print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%")
        return
    
    args.output_dir = get_next_exp_dir(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    max_accuracy = 0.0
    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    training_history = []

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print("tqdm not installed, using simple progress display")

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        
        if use_tqdm:
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler, 
            args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer,
            args=args
        )
        
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        if data_loader_val is not None:
            test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp, num_classes=args.nb_classes)
            
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
            
            print(f'Max accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats.update({**{f'test_{k}': v for k, v in test_stats.items()}})
            
            if args.model_ema and args.model_ema_eval:
                test_stats_ema = evaluate(data_loader_val, model_ema.ema, device, use_amp=args.use_amp, num_classes=args.nb_classes)
                if max_accuracy_ema < test_stats_ema["acc1"]:
                    max_accuracy_ema = test_stats_ema["acc1"]
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch="best-ema", model_ema=model_ema)
                    print(f'Max EMA accuracy: {max_accuracy_ema:.2f}%')
                if log_writer is not None:
                    log_writer.update(test_acc1_ema=test_stats_ema['acc1'], head="perf", step=epoch)
                log_stats.update({**{f'test_{k}_ema': v for k, v in test_stats_ema.items()}})
        
        training_history.append(log_stats)
        
        if args.output_dir and utils.is_main_process():
            os.makedirs(args.output_dir, exist_ok=True)
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            save_results_csv(args.output_dir, training_history)
            plot_results(args.output_dir, training_history, num_classes=args.nb_classes)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'\nTraining complete in {total_time_str}')
    print(f'Max accuracy: {max_accuracy:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('FCMAE fine-tuning', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
