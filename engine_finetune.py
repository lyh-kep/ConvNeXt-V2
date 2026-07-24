import math
from typing import Iterable, Optional

import torch
import numpy as np
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import ModelEma

from torchmetrics import Accuracy, Precision, Recall,  Specificity,F1
from sklearn.metrics import roc_curve, auc

import utils
from utils import adjust_learning_rate

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, 
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = max(1, len(data_loader) // 5)

    update_freq = args.update_freq
    use_amp = args.use_amp
    optimizer.zero_grad()

    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % update_freq == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else:
            output = model(samples)
            loss = criterion(output, targets)

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

        if mixup_fn is None:
            preds = output.max(-1)[-1]
            class_acc = (preds == targets).float().mean()
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        else:
            class_acc = None

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()
    
    metric_logger.synchronize_between_processes()
    
    if total_samples > 0:
        train_acc = total_correct / total_samples
        print(f"\n=== Training Stats - Epoch {epoch} ===")
        print(f"  Total Samples: {total_samples}")
        print(f"  Correct: {total_correct}")
        print(f"  Accuracy: {train_acc:.4f}")
        print(f"  Loss: {metric_logger.loss.global_avg:.4f}")
        print(f"  Avg LR: {metric_logger.lr.global_avg:.6f}")
        print("=" * 50)
    
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader: Iterable, model: torch.nn.Module, device: torch.device, use_amp=False, num_classes=2):
    criterion = torch.nn.CrossEntropyLoss()
    
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'


    accuracy_metric = Accuracy( num_classes=num_classes).to(device)
    precision_metric = Precision( num_classes=num_classes, average='macro').to(device)
    recall_metric = Recall( num_classes=num_classes, average='macro').to(device)
    f1_metric = F1(num_classes=num_classes, average='macro').to(device)
    specificity_metric = Specificity( num_classes=num_classes, average='macro').to(device)

    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    all_probs = []

    for batch in metric_logger.log_every(data_loader, 10, header):
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
        
        preds_acc = output.max(-1)[-1]
        acc1 = (preds_acc == target).float().mean() * 100.0
        
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

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    
    acc = accuracy_metric.compute().item()
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    f1 = f1_metric.compute().item()
    specificity = specificity_metric.compute().item()
    
    print(f"\n=== Evaluation Stats ===")
    print(f"  Total Samples: {total_samples}")
    print(f"  Correct: {total_correct}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Loss: {metric_logger.loss.global_avg:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    
    if total_samples > 0:
        num_classes_actual = len(set(all_targets))
        conf_matrix = compute_confusion_matrix(all_preds, all_targets, num_classes_actual)
        print(f"\n  Confusion Matrix ({num_classes_actual}x{num_classes_actual}):")
        for i in range(num_classes_actual):
            row = "    "
            for j in range(num_classes_actual):
                row += f"{conf_matrix[i][j]:4d} "
            print(row)
    
    print("=" * 50)
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    result = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
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



