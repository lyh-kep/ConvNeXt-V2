import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
import csv
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageDraw
import models.convnextv2 as convnextv2
from utils import remap_checkpoint_keys, load_state_dict

def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt V2 Inference', add_help=False)
    parser.add_argument('--model', default='convnextv2_atto', type=str,
                        help='模型名称，支持: convnextv2_atto, convnextv2_femto, convnextv2_nano, '
                             'convnextv2_tiny, convnextv2_base, convnextv2_large, convnextv2_huge')
    parser.add_argument('--input_size', default=224, type=int,
                        help='输入图像尺寸')
    parser.add_argument('--checkpoint', default='weights/convnextv2_atto_1k_224_fcmae.pt', type=str,
                        help='模型权重路径')
    parser.add_argument('--img_path', default='test.jpg', type=str,
                        help='输入图像路径，可以是单张图片(.jpg/.png)或文件夹路径')
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='类别数量，二分类设为2')
    parser.add_argument('--class_names', default='cat,dog', type=str,
                        help='类别名称，逗号分隔，如: cat,dog')
    parser.add_argument('--output_dir', default='runs/detect/exp', type=str,
                        help='输出目录，自动创建exp/exp2/exp3等子目录')
    parser.add_argument('--save_images', action='store_true',
                        help='是否保存带预测标签的图片')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='二分类阈值，大于此值为正类')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='批量推理大小')
    return parser

def load_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = convnextv2.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=0.0,
        head_init_scale=1.0,
    )
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    state_dict = model.state_dict()
    for k in list(checkpoint_model.keys()):
        if k in state_dict:
            if checkpoint_model[k].shape != state_dict[k].shape:
                if 'grn' in k:
                    checkpoint_model[k] = checkpoint_model[k].view(state_dict[k].shape)
                elif k in ['head.weight', 'head.bias']:
                    del checkpoint_model[k]
                else:
                    del checkpoint_model[k]
        else:
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

def predict_single(model, img_path, transform, device, class_names, threshold=0.5):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
    
    prob_np = probabilities.cpu().numpy()[0]
    pred_idx = prob_np.argmax()
    
    if len(class_names) == 2:
        pos_prob = prob_np[1]
        is_positive = pos_prob >= threshold
        return {
            'image': img_path,
            'prediction': class_names[pred_idx],
            'prediction_idx': int(pred_idx),
            'probability': float(prob_np[pred_idx]),
            'positive_probability': float(pos_prob),
            'is_positive': bool(is_positive),
            'all_probabilities': [float(p) for p in prob_np],
        }
    else:
        return {
            'image': img_path,
            'prediction': class_names[pred_idx],
            'prediction_idx': int(pred_idx),
            'probability': float(prob_np[pred_idx]),
            'all_probabilities': [float(p) for p in prob_np],
        }

def predict_batch(model, img_paths, transform, device, class_names, batch_size=1, threshold=0.5):
    results = []
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
        
        prob_np = probabilities.cpu().numpy()
        
        for j, path in enumerate(batch_paths):
            pred_idx = prob_np[j].argmax()
            
            if len(class_names) == 2:
                pos_prob = prob_np[j][1]
                is_positive = pos_prob >= threshold
                results.append({
                    'image': path,
                    'prediction': class_names[pred_idx],
                    'prediction_idx': int(pred_idx),
                    'probability': float(prob_np[j][pred_idx]),
                    'positive_probability': float(pos_prob),
                    'is_positive': bool(is_positive),
                    'all_probabilities': [float(p) for p in prob_np[j]],
                })
            else:
                results.append({
                    'image': path,
                    'prediction': class_names[pred_idx],
                    'prediction_idx': int(pred_idx),
                    'probability': float(prob_np[j][pred_idx]),
                    'all_probabilities': [float(p) for p in prob_np[j]],
                })
    
    return results

def save_results_to_csv(results, csv_path):
    if not results:
        return
    
    fieldnames = list(results[0].keys())
    fieldnames.remove('all_probabilities')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: v for k, v in r.items() if k != 'all_probabilities'}
            writer.writerow(row)
    
    print(f"Results saved to {csv_path}")

def save_image_with_prediction(img_path, result, class_names, output_dir):
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    text = f"{result['prediction']}: {result['probability']:.4f}"
    if len(class_names) == 2:
        text += f" (Pos: {result['positive_probability']:.4f})"
    
    draw.text((10, 10), text, fill=(255, 0, 0))
    
    rel_path = os.path.relpath(img_path)
    output_path = os.path.join(output_dir, rel_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)

def get_output_dir(base_dir):
    if os.path.exists(base_dir):
        i = 2
        while os.path.exists(base_dir + str(i)):
            i += 1
        return base_dir + str(i)
    return base_dir

def main(args):
    class_names = args.class_names.split(',')
    if len(class_names) != args.nb_classes:
        print(f"Warning: class_names has {len(class_names)} classes but nb_classes={args.nb_classes}")
        class_names = [f'class_{i}' for i in range(args.nb_classes)]
    
    output_dir = get_output_dir(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    model, device = load_model(args)
    transform = get_transform(args)
    
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Input size: {args.input_size}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    if os.path.isdir(args.img_path):
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        img_paths = []
        
        for root, dirs, files in os.walk(args.img_path):
            for f in files:
                if os.path.splitext(f)[1].lower() in img_extensions:
                    img_paths.append(os.path.join(root, f))
        
        img_paths = sorted(img_paths)
        
        if not img_paths:
            print(f"No images found in {args.img_path}")
            return
        
        print(f"Found {len(img_paths)} images in directory")
        results = predict_batch(model, img_paths, transform, device, class_names, 
                               batch_size=args.batch_size, threshold=args.threshold)
        
        for r in results:
            rel_path = os.path.relpath(r['image'], args.img_path)
            if len(class_names) == 2:
                print(f"{rel_path:30s} -> {r['prediction']:10s} ({r['probability']:.4f})  Positive: {r['positive_probability']:.4f}")
            else:
                print(f"{rel_path:30s} -> {r['prediction']:10s} ({r['probability']:.4f})")
            
            if args.save_images:
                save_image_with_prediction(r['image'], r, class_names, output_dir)
        
        csv_path = os.path.join(output_dir, 'results.csv')
        save_results_to_csv(results, csv_path)
        
        if len(class_names) == 2:
            positive_count = sum(1 for r in results if r['is_positive'])
            print(f"\nTotal: {len(results)} | Positive: {positive_count} | Negative: {len(results) - positive_count}")
    
    else:
        result = predict_single(model, args.img_path, transform, device, class_names, threshold=args.threshold)
        
        print(f"\nImage: {result['image']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['probability']:.4f}")
        
        if args.save_images:
            save_image_with_prediction(args.img_path, result, class_names, output_dir)
        
        csv_path = os.path.join(output_dir, 'results.csv')
        save_results_to_csv([result], csv_path)
        
        if len(class_names) == 2:
            print(f"Positive probability: {result['positive_probability']:.4f}")
            print(f"Is positive: {result['is_positive']}")
        
        print("\nClass probabilities:")
        for name, prob in zip(class_names, result['all_probabilities']):
            print(f"  {name}: {prob:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt V2 Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

