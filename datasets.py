# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        if is_train:
            root = args.data_path
        else:
            if args.eval_data_path is not None:
                root = args.eval_data_path
            elif os.path.basename(args.data_path) == 'train':
                root = os.path.join(os.path.dirname(args.data_path), 'val')
            elif os.path.basename(args.data_path) == 'val':
                root = args.data_path
            else:
                root = os.path.join(args.data_path, 'val')

        if not os.path.exists(root):
            raise FileNotFoundError(
                f"\n{'='*70}\n"
                f"  [ERROR] {'训练集' if is_train else '验证集'}路径不存在: {root}\n"
                f"  请确认路径正确，或使用命令行参数指定：\n"
                f"    --data_path /path/to/train  --eval_data_path /path/to/val\n"
                f"  数据目录结构应为：\n"
                f"    dataset/train/class1/*.jpg\n"
                f"    dataset/train/class2/*.jpg\n"
                f"    dataset/val/class1/*.jpg\n"
                f"    dataset/val/class2/*.jpg\n"
                f"{'='*70}"
            )

        dataset = datasets.ImageFolder(root, transform=transform)
        actual_nb_classes = len(dataset.class_to_idx)

        if actual_nb_classes == 0:
            raise ValueError(
                f"\n{'='*70}\n"
                f"  [ERROR] {'训练集' if is_train else '验证集'}中没有检测到任何类别子目录！\n"
                f"  路径: {root}\n"
                f"  请确认子目录结构是否正确（每个类别一个子文件夹）。\n"
                f"{'='*70}"
            )

        if getattr(args, '_train_classes', None) is None and is_train:
            args._train_classes = dataset.classes
            args._train_class_to_idx = dataset.class_to_idx

        if not is_train and hasattr(args, '_train_classes') and args._train_classes is not None:
            if dataset.classes != args._train_classes:
                raise ValueError(
                    f"\n{'='*70}\n"
                    f"  [ERROR] 训练集和验证集的类别不一致！\n"
                    f"  训练集类别 ({len(args._train_classes)}): {args._train_classes}\n"
                    f"  验证集类别 ({len(dataset.classes)}): {dataset.classes}\n"
                    f"  请确认 train/ 和 val/ 目录下的子目录名称完全一致。\n"
                    f"{'='*70}"
                )

        if args.nb_classes != actual_nb_classes:
            print(f"[INFO] 检测到实际类别数量: {actual_nb_classes}, 自动更新 args.nb_classes (原: {args.nb_classes})")
            args.nb_classes = actual_nb_classes

        nb_classes = args.nb_classes
    else:
        raise NotImplementedError()

    print(f"Number of the class = {nb_classes}")
    print(f"Data root: {root if args.data_set == 'image_folder' else args.data_path}")
    if hasattr(dataset, 'classes'):
        print(f"Class names: {dataset.classes}")
        print(f"Samples count: {len(dataset)}")
    print("---------------------------")

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.input_size >= 384:
            t.append(
            transforms.Resize((args.input_size, args.input_size),
                            interpolation=transforms.InterpolationMode.BICUBIC),
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
