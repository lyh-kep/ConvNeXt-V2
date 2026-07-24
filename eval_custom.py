import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import warnings
warnings.filterwarnings('ignore', message='Argument interpolation should be of type InterpolationMode')
warnings.filterwarnings('ignore', message='Default upsampling behavior')

import argparse
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import models.convnextv2 as convnextv2
from utils import remap_checkpoint_keys, load_state_dict

def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt V2 Evaluation', add_help=False)
    parser.add_argument('--model', default='convnextv2_atto', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--checkpoint', default='weights/convnextv2_atto_1k_224_fcmae.pt', type=str)
    parser.add_argument('--data_path', default='dataset/val', type=str, help='Validation data path (folder with class subfolders)')
    parser.add_argument('--nb_classes', default=2, type=int)
    parser.add_argument('--class_names', default='cat,dog', type=str, help='Comma-separated class names')
    parser.add_argument('--output_dir', default='eval_results', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--threshold', default=0.5, type=float)
    return parser

def load_model(args):
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            "\n" + "="*70 + "\n"
            "  [ERROR] 权重文件不存在: " + args.checkpoint + "\n"
            "  请使用 --checkpoint /path/to/weights.pt 指定正确路径。\n"
            + "="*70
        )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = convnextv2.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=0.0,
        head_init_scale=1.0,
    )

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
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
    load_state_dict(model, checkpoint_model, prefix='')

    model.to(device)
    model.eval()
    return model, device

def get_transform(args):
    return transforms.Compose([
        transforms.Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def load_data(args):
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    class_names = args.class_names.split(',')
    if len(class_names) != args.nb_classes:
        class_names = sorted([d for d in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, d))])
        args.nb_classes = len(class_names)

    img_paths = []
    labels = []

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(args.data_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        for f in os.listdir(class_dir):
            if os.path.splitext(f)[1].lower() in img_extensions:
                img_paths.append(os.path.join(class_dir, f))
                labels.append(class_idx)

    return img_paths, labels, class_names

def evaluate_model(model, img_paths, labels, transform, device, batch_size=8):
    all_preds = []
    all_probs = []

    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i+batch_size]
        batch_imgs = []
        for path in batch_paths:
            img = Image.open(path).convert('RGB')
            batch_imgs.append(transform(img))

        batch_tensor = torch.stack(batch_imgs).to(device)

        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = F.softmax(outputs, dim=1)

        all_preds.extend(probabilities.argmax(dim=1).cpu().numpy())
        all_probs.extend(probabilities.cpu().numpy())

    return np.array(all_preds), np.array(all_probs)

def compute_metrics(labels, preds, probs, class_names, threshold=0.5):
    metrics = {}

    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['precision'] = precision_score(labels, preds, average='macro', zero_division=1)
    metrics['recall'] = recall_score(labels, preds, average='macro', zero_division=1)
    metrics['f1'] = f1_score(labels, preds, average='macro', zero_division=1)

    if len(class_names) == 2:
        metrics['precision_binary'] = precision_score(labels, preds, pos_label=1, zero_division=1)
        metrics['recall_binary'] = recall_score(labels, preds, pos_label=1, zero_division=1)
        metrics['f1_binary'] = f1_score(labels, preds, pos_label=1, zero_division=1)

        if len(set(labels)) == 2:
            metrics['roc_auc'] = roc_auc_score(labels, probs[:, 1])

            fpr, tpr, _ = roc_curve(labels, probs[:, 1])
            metrics['roc_fpr'] = fpr
            metrics['roc_tpr'] = tpr

            precision, recall, _ = precision_recall_curve(labels, probs[:, 1])
            metrics['pr_auc'] = auc(recall, precision)
            metrics['pr_precision'] = precision
            metrics['pr_recall'] = recall

    metrics['confusion_matrix'] = confusion_matrix(labels, preds)
    metrics['classification_report'] = classification_report(labels, preds, target_names=class_names, output_dict=True, zero_division=1)

    return metrics

def plot_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, output_path):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_pr_curve(precision, recall, pr_auc, output_path):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_metrics_to_csv(metrics, class_names, output_path):
    rows = []

    rows.append({'metric': 'accuracy', 'value': metrics['accuracy']})
    rows.append({'metric': 'precision_macro', 'value': metrics['precision']})
    rows.append({'metric': 'recall_macro', 'value': metrics['recall']})
    rows.append({'metric': 'f1_macro', 'value': metrics['f1']})

    if 'precision_binary' in metrics:
        rows.append({'metric': 'precision_binary', 'value': metrics['precision_binary']})
        rows.append({'metric': 'recall_binary', 'value': metrics['recall_binary']})
        rows.append({'metric': 'f1_binary', 'value': metrics['f1_binary']})

    if 'roc_auc' in metrics:
        rows.append({'metric': 'roc_auc', 'value': metrics['roc_auc']})
        rows.append({'metric': 'pr_auc', 'value': metrics['pr_auc']})

    for class_name in class_names:
        if class_name in metrics['classification_report']:
            report = metrics['classification_report'][class_name]
            rows.append({'metric': f'{class_name}_precision', 'value': report['precision']})
            rows.append({'metric': f'{class_name}_recall', 'value': report['recall']})
            rows.append({'metric': f'{class_name}_f1', 'value': report['f1-score']})
            rows.append({'metric': f'{class_name}_support', 'value': report['support']})

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['metric', 'value'])
        writer.writeheader()
        writer.writerows(rows)

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    class_names = args.class_names.split(',')

    print(f"Loading model: {args.model}")
    model, device = load_model(args)

    print(f"Loading data from: {args.data_path}")
    img_paths, labels, class_names = load_data(args)
    args.class_names = ','.join(class_names)

    print(f"Found {len(img_paths)} images across {len(class_names)} classes")
    print(f"Classes: {class_names}")

    transform = get_transform(args)

    print(f"\nEvaluating on {device}...")
    preds, probs = evaluate_model(model, img_paths, labels, transform, device, args.batch_size)

    print(f"\nComputing metrics...")
    metrics = compute_metrics(labels, preds, probs, class_names, args.threshold)

    print("\n" + "=" * 60)
    print("Evaluation Report")
    print("=" * 60)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision']:.4f}")
    print(f"Recall (macro): {metrics['recall']:.4f}")
    print(f"F1 (macro): {metrics['f1']:.4f}")

    if len(class_names) == 2:
        print(f"\nBinary Classification Metrics:")
        print(f"Precision: {metrics['precision_binary']:.4f}")
        print(f"Recall: {metrics['recall_binary']:.4f}")
        print(f"F1: {metrics['f1_binary']:.4f}")
        if 'roc_auc' in metrics:
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
            print(f"PR AUC: {metrics['pr_auc']:.4f}")

    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])

    print(f"\nClassification Report:")
    print(classification_report(labels, preds, target_names=class_names, zero_division=1))

    print(f"\nSaving results to: {args.output_dir}")

    plot_confusion_matrix(metrics['confusion_matrix'], class_names,
                          os.path.join(args.output_dir, 'confusion_matrix.png'))

    if len(class_names) == 2 and 'roc_auc' in metrics:
        plot_roc_curve(metrics['roc_fpr'], metrics['roc_tpr'], metrics['roc_auc'],
                       os.path.join(args.output_dir, 'roc_curve.png'))
        plot_pr_curve(metrics['pr_precision'], metrics['pr_recall'], metrics['pr_auc'],
                      os.path.join(args.output_dir, 'pr_curve.png'))

    save_metrics_to_csv(metrics, class_names,
                        os.path.join(args.output_dir, 'metrics.csv'))

    with open(os.path.join(args.output_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Data Path: {args.data_path}\n")
        f.write(f"Input Size: {args.input_size}\n")
        f.write(f"Classes: {class_names}\n")
        f.write(f"Total Samples: {len(img_paths)}\n\n")

        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision (macro): {metrics['precision']:.4f}\n")
        f.write(f"Recall (macro): {metrics['recall']:.4f}\n")
        f.write(f"F1 (macro): {metrics['f1']:.4f}\n\n")

        if len(class_names) == 2:
            f.write("Binary Classification Metrics:\n")
            f.write(f"Precision: {metrics['precision_binary']:.4f}\n")
            f.write(f"Recall: {metrics['recall_binary']:.4f}\n")
            f.write(f"F1: {metrics['f1_binary']:.4f}\n")
            if 'roc_auc' in metrics:
                f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
                f.write(f"PR AUC: {metrics['pr_auc']:.4f}\n")
            f.write("\n")

        f.write("Confusion Matrix:\n")
        f.write(str(metrics['confusion_matrix']) + "\n\n")

        f.write("Classification Report:\n")
        f.write(classification_report(labels, preds, target_names=class_names, zero_division=1))

    print("\nEvaluation complete!")
    print(f"Results saved to: {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt V2 Evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
