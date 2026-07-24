# ConvNeXt-V2 图像分类

基于 ConvNeXt-V2 官方实现的图像分类项目，支持训练、评估和推理。

## 环境要求

```bash
# Python 3.x
# PyTorch 1.8.1+cu111
# torchvision 0.9.1+cu111
# timm 0.3.2
# torchmetrics 0.11.4
# matplotlib 3.10.3
# scikit-learn 1.6.1
# tqdm 4.66.5
# setuptools 80.9.0
```

## 安装依赖

```bash
pip install torch torchvision torchmetrics timm matplotlib scikit-learn tqdm
```

## 项目结构

```
ConvNeXt-V2-main/
├── models/                 # 模型定义
│   ├── convnextv2.py       # ConvNeXt-V2 模型
│   ├── fcmae.py            # FCMAE 预训练模型
│   └── utils.py            # 模型工具函数
├── weights/                # 预训练权重
│   └── convnextv2_atto_1k_224_fcmae.pt
├── dataset/                # 数据集
│   ├── train/
│   │   ├── class1/
│   │   └── class2/
│   └── val/
│       ├── class1/
│       └── class2/
├── runs/                   # 输出目录
│   ├── train/              # 训练输出
│   │   ├── exp/
│   │   ├── exp2/
│   │   └── ...
│   └── detect/             # 推理输出
│       ├── exp/
│       ├── exp2/
│       └── ...
├── train_custom.py         # 训练脚本
├── eval_custom.py          # 评估脚本
├── inference.py            # 推理脚本
├── utils.py                # 通用工具函数
├── optim_factory.py        # 优化器工厂
└── datasets.py             # 数据集加载
```

## 支持的模型

| 模型名称 | 参数数量 | 适用场景 |
|---------|---------|---------|
| convnextv2_atto | ~3.3M | 轻量模型，适合小数据集 |
| convnextv2_femto | ~5.1M | 轻量模型 |
| convnextv2_nano | ~7.0M | 轻量模型 |
| convnextv2_tiny | ~15.6M | 中等模型 |
| convnextv2_base | ~86.5M | 标准模型 |
| convnextv2_large | ~198M | 大型模型 |
| convnextv2_huge | ~350M | 超大型模型 |

## 数据集格式

```
dataset/
├── train/
│   ├── cat/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── dog/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── val/
    ├── cat/
    └── dog/
```

## 训练

### 基础训练

```bash
python train_custom.py \
--data_path "dataset/train" \
--model convnextv2_atto \
--finetune "weights/convnextv2_atto_1k_224_fcmae.pt" \
--nb_classes 2 \
--epochs 100 \
--batch_size 8
```

### 常用参数

```bash
python train_custom.py \
--data_path "dataset/train" \          # 训练数据集路径
--eval_data_path "dataset/val" \       # 验证数据集路径（可选）
--nb_classes 2 \                       # 类别数量
--model convnextv2_atto \              # 模型名称
--input_size 224 \                     # 输入图像尺寸
--finetune "weights/convnextv2_atto_1k_224_fcmae.pt" \
--batch_size 8 \                       # 批大小
--epochs 100 \                         # 训练轮数
--blr 2e-4 \                           # 基础学习率
--layer_decay 0.9 \                    # 分层学习率衰减
--output_dir "runs/train/" \           # 输出目录
--device "cuda" \                      # 训练设备
--num_workers 0 \                      # 数据加载线程数（Windows建议0）
--use_amp False                        # 是否使用混合精度
```

### 多分类训练

```bash
python train_custom.py \
--data_path "dataset/train" \
--model convnextv2_base \
--finetune "weights/convnextv2_base_1k_224_fcmae.pt" \
--nb_classes 10 \                       # 10分类
--epochs 100 \
--batch_size 4
```

## 评估

### 评估训练好的模型

```bash
python eval_custom.py \
--data_path "dataset/val" \
--checkpoint "runs/train/exp/checkpoint-best.pth" \
--nb_classes 2 \
--class_names "cat,dog"
```

### 评估输出

- `metrics.csv` - 所有评估指标
- `report.txt` - 完整评估报告
- `confusion_matrix.png` - 混淆矩阵图
- `roc_curve.png` - ROC曲线（二分类）
- `pr_curve.png` - PR曲线（二分类）

## 推理

### 单张图片推理

```bash
python inference.py \
--img_path "test.jpg" \
--checkpoint "runs/train/exp/checkpoint-best.pth" \
--nb_classes 2 \
--class_names "cat,dog"
```

### 批量推理（文件夹）

```bash
python inference.py \
--img_path "dataset/val" \
--checkpoint "runs/train/exp/checkpoint-best.pth" \
--nb_classes 2 \
--class_names "cat,dog" \
--save_images                         # 保存带预测标签的图片
```

### 推理输出

- `results.csv` - 推理结果
- 带预测标签的图片（当指定 `--save_images` 时）

## 训练输出

训练完成后，`runs/train/exp*` 目录包含：

```
runs/train/exp/
├── checkpoint-best.pth          # 最佳模型权重
├── checkpoint-best-ema.pth      # EMA最佳模型权重
├── checkpoint-{epoch}.pth       # 各epoch模型权重
├── results.csv                  # 训练指标（每epoch）
├── results.png                  # 训练曲线
└── log.txt                      # 详细日志
```

### results.csv 包含的指标

| 指标 | 说明 |
|------|------|
| epoch | 训练轮数 |
| train_loss | 训练损失 |
| train_class_acc | 训练准确率 |
| train_lr | 学习率 |
| test_loss | 测试损失 |
| test_acc1 | 测试 Top-1 准确率 |
| test_acc5 | 测试 Top-5 准确率 |
| test_acc1_ema | EMA 模型测试准确率 |

## 评估指标

`eval_custom.py` 输出的指标：

- **基础指标**: Accuracy, Precision, Recall, F1
- **二分类指标**: ROC AUC, PR AUC
- **混淆矩阵**: 可视化 + 数值
- **分类报告**: 每个类别的 precision/recall/f1/support

## 注意事项

1. **数据集**: 确保数据集按文件夹组织，每个类别一个子目录
2. **预训练权重**: 使用 FCMAE 预训练权重时，分类头会被随机初始化并重新训练
3. **学习率**: 建议根据 batch_size 调整 `--blr` 参数
4. **Windows 用户**: `--num_workers` 建议设为 0
5. **MinkowskiEngine**: 仅在 FCMAE 预训练阶段需要，推理和微调不需要

## 示例

### 完整训练流程

```bash
# 训练
python train_custom.py \
--data_path "dataset/train" \
--model convnextv2_atto \
--finetune "weights/convnextv2_atto_1k_224_fcmae.pt" \
--nb_classes 2 \
--epochs 100 \
--batch_size 8

# 评估
python eval_custom.py \
--data_path "dataset/val" \
--checkpoint "runs/train/exp/checkpoint-best.pth" \
--nb_classes 2 \
--class_names "cat,dog"

# 推理
python inference.py \
--img_path "dataset/val" \
--checkpoint "runs/train/exp/checkpoint-best.pth" \
--nb_classes 2 \
--class_names "cat,dog" \
--save_images
```
