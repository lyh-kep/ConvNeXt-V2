import math
import sys
from typing import Iterable, Optional
import numpy as np

import torch
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import ModelEma

from torchmetrics import Accuracy, Precision, Recall, Specificity, F1
from sklearn.metrics import roc_curve, auc

import utils
from utils import adjust_learning_rate

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    log_writer=None, args=None, total_epochs=None):
    model.train(True)
    update_freq = args.update_freq
    use_amp = args.use_amp
    optimizer.zero_grad()

    num_classes = getattr(args, 'nb_classes', 2)

    accuracy_metric = Accuracy(num_classes=num_classes).to(device)
    precision_metric = Precision(num_classes=num_classes, average='macro').to(device)
    recall_metric = Recall(num_classes=num_classes, average='macro').to(device)
    f1_metric = F1(num_classes=num_classes, average='macro').to(device)

    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    n_steps = 0
    all_preds = []
    all_targets = []

    if HAS_TQDM:
        if total_epochs is not None:
            desc_prefix = f'[{epoch+1}/{total_epochs}] '
        else:
            desc_prefix = f'Epoch [{epoch}] '
        process_bar = tqdm(data_loader, file=sys.stdout, desc=desc_prefix + 'Initializing...')
    else:
        process_bar = data_loader

    for data_iter_step, (samples, targets) in enumerate(process_bar):
        if data_iter_step % update_freq == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets_mixup = mixup_fn(samples, targets)
        else:
            targets_mixup = targets

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets_mixup)
        else:
            output = model(samples)
            loss = criterion(output, targets_mixup)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else:
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        total_loss += loss_value
        n_steps += 1

        if mixup_fn is None:
            preds = output.max(-1)[-1]
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

            accuracy_metric.update(preds, targets)
            precision_metric.update(preds, targets)
            recall_metric.update(preds, targets)
            f1_metric.update(preds, targets)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            cur_acc = accuracy_metric.compute().item()
            cur_pre = precision_metric.compute().item()
            cur_rec = recall_metric.compute().item()
            cur_f1 = f1_metric.compute().item()
            cur_loss = total_loss / n_steps

            if HAS_TQDM:
                max_lr = 0.
                for group in optimizer.param_groups:
                    max_lr = max(max_lr, group["lr"])
                process_bar.set_description(
                    f'{desc_prefix}TrainAcc: {cur_acc:.4f} | TrainLoss: {cur_loss:.4f} | '
                    f'Recall: {cur_rec:.4f} | Pre: {cur_pre:.4f} | F1: {cur_f1:.4f} | LR: {max_lr:.6f}'
                )
        else:
            if HAS_TQDM:
                max_lr = 0.
                for group in optimizer.param_groups:
                    max_lr = max(max_lr, group["lr"])
                cur_loss = total_loss / n_steps
                process_bar.set_description(
                    f'{desc_prefix}TrainLoss: {cur_loss:.4f} | LR: {max_lr:.6f} (mixup on)'
                )

    if HAS_TQDM:
        process_bar.close()

    min_lr = 10.
    max_lr = 0.
    for group in optimizer.param_groups:
        min_lr = min(min_lr, group["lr"])
        max_lr = max(max_lr, group["lr"])

    avg_loss = total_loss / max(n_steps, 1)

    result = {
        'loss': avg_loss,
        'lr': max_lr,
        'min_lr': min_lr,
    }

    result['n_steps'] = n_steps
    result['n_samples'] = total_samples

    if n_steps == 0:
        print(f'  [WARN] train dataloader produced 0 batches! Check --data_path and dataset directory structure.')

    if mixup_fn is None and total_samples > 0:
        train_acc = total_correct / total_samples
        result['acc'] = train_acc * 100.0
        result['precision'] = precision_metric.compute().item()
        result['recall'] = recall_metric.compute().item()
        result['f1'] = f1_metric.compute().item()

        print(f'  Train -> Acc: {result["acc"]:.2f}% | Loss: {avg_loss:.4f} | '
              f'P: {result["precision"]:.4f} | R: {result["recall"]:.4f} | F1: {result["f1"]:.4f} | '
              f'Samples: {total_samples}')
    elif mixup_fn is None:
        result['acc'] = 0.0
        result['precision'] = 0.0
        result['recall'] = 0.0
        result['f1'] = 0.0
        print(f'  Train -> Acc: N/A (no samples) | Loss: {avg_loss:.4f} | LR: {max_lr:.6f}')
    else:
        result['acc'] = 0.0
        result['precision'] = 0.0
        result['recall'] = 0.0
        result['f1'] = 0.0
        print(f'  Train -> Loss: {avg_loss:.4f} | LR: {max_lr:.6f} (mixup/cutmix active)')

    return result


@torch.no_grad()
def evaluate(data_loader: Iterable, model: torch.nn.Module, device: torch.device, use_amp=False, num_classes=2, epoch=None, total_epochs=None):
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()

    accuracy_metric = Accuracy(num_classes=num_classes).to(device)
    precision_metric = Precision(num_classes=num_classes, average='macro').to(device)
    recall_metric = Recall(num_classes=num_classes, average='macro').to(device)
    f1_metric = F1(num_classes=num_classes, average='macro').to(device)
    specificity_metric = Specificity(num_classes=num_classes, average='macro').to(device)

    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    n_steps = 0
    all_preds = []
    all_targets = []
    all_probs = []

    if HAS_TQDM:
        if epoch is not None and total_epochs is not None:
            desc_prefix = f'[{epoch+1}/{total_epochs}] '
        else:
            desc_prefix = ''
        process_bar = tqdm(data_loader, file=sys.stdout, desc=desc_prefix + 'Val Initializing...')
    else:
        process_bar = data_loader

    for batch in process_bar:
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                if isinstance(output, dict):
                    output = output['logits']
                loss = criterion(output, target)
        else:
            output = model(images)
            if isinstance(output, dict):
                output = output['logits']
            loss = criterion(output, target)

        torch.cuda.synchronize()

        total_loss += loss.item()
        n_steps += 1

        preds = output.max(-1)[-1]
        probs = F.softmax(output, dim=1)

        accuracy_metric.update(preds, target)
        precision_metric.update(preds, target)
        recall_metric.update(preds, target)
        f1_metric.update(preds, target)
        specificity_metric.update(preds, target)

        total_correct += (preds == target).sum().item()
        total_samples += target.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        cur_acc = accuracy_metric.compute().item()
        cur_pre = precision_metric.compute().item()
        cur_rec = recall_metric.compute().item()
        cur_f1 = f1_metric.compute().item()
        cur_spe = specificity_metric.compute().item()
        cur_loss = total_loss / n_steps

        if HAS_TQDM:
            process_bar.set_description(
                f'{desc_prefix}ValAcc: {cur_acc:.4f} | ValLoss: {cur_loss:.4f} | '
                f'Recall: {cur_rec:.4f} | Spe: {cur_spe:.4f} | Pre: {cur_pre:.4f} | F1: {cur_f1:.4f}'
            )

    if HAS_TQDM:
        process_bar.close()

    avg_loss = total_loss / max(n_steps, 1)
    acc = accuracy_metric.compute().item()
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    f1 = f1_metric.compute().item()
    specificity = specificity_metric.compute().item()

    acc1_percent = (total_correct / max(total_samples, 1)) * 100.0

    if n_steps == 0:
        print(f'  [WARN] val dataloader produced 0 batches! Check --eval_data_path / --data_path.')

    print(f'  Val   -> Acc: {acc1_percent:.2f}% | Loss: {avg_loss:.4f} | '
          f'P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f} | Spe: {specificity:.4f} | Samples: {total_samples}')

    result = {
        'loss': avg_loss,
        'acc': acc1_percent,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'n_steps': n_steps,
        'n_samples': total_samples,
    }

    if num_classes == 2 and total_samples > 0 and len(set(all_targets)) == 2:
        all_probs_np = np.array(all_probs)
        if all_probs_np.ndim == 2 and all_probs_np.shape[1] == 2:
            y_score = all_probs_np[:, 1]
        else:
            y_score = all_probs_np.ravel()

        fpr, tpr, _ = roc_curve(all_targets, y_score)
        roc_auc = auc(fpr, tpr)

        result['fpr'] = fpr.tolist()
        result['tpr'] = tpr.tolist()
        result['roc_auc'] = roc_auc

    return result


def compute_confusion_matrix(preds, targets, num_classes):
    conf_matrix = [[0] * num_classes for _ in range(num_classes)]
    for p, t in zip(preds, targets):
        conf_matrix[t][p] += 1
    return conf_matrix


def compute_binary_metrics(preds, targets):
    tp = sum(1 for p, t in zip(preds, targets) if p == 1 and t == 1)
    tn = sum(1 for p, t in zip(preds, targets) if p == 0 and t == 0)
    fp = sum(1 for p, t in zip(preds, targets) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(preds, targets) if p == 0 and t == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1
