import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
import argparse

# 导入 SAM 相关模块
try:
    from model.SAMframe import (
        sam,
        mask_generator,
        matcher,
        RegionHead,
        SAMFrameExtractor,
        extract_region_features,
        find_unmatched_regions
    )
    from segment_anything import SamPredictor
    import torchvision.ops as tv_ops
except ImportError:
    print("Error: Cannot import from model.SAMframe. Please check your path.")
    exit(1)

def filter_masks(masks_data):
    """筛选高质量掩码"""
    min_area = 100
    iou_thresh = 0.8
    valid_masks = [m for m in masks_data if m['predicted_iou'] > iou_thresh and m['area'] > min_area]
    
    if len(valid_masks) > 0:
        boxes_orig = [m['bbox'] for m in valid_masks]
        scores = [m['predicted_iou'] for m in valid_masks]
        boxes_xyxy = []
        for b in boxes_orig:
            x, y, w, h = b
            boxes_xyxy.append([x, y, x + w, y + h])
        boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        keep = tv_ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
        valid_masks = [valid_masks[i] for i in keep]
    return valid_masks

def process_masks(S_feat, valid_masks, orig_h, orig_w, device):
    """处理掩码和边界框"""
    _, h_feat, w_feat = S_feat.shape
    scale_h = h_feat / orig_h
    scale_w = w_feat / orig_w
    
    masks_down = []
    boxes_feat = []
    
    for m in valid_masks:
        mask_orig = m['segmentation']
        mask_resized = cv2.resize(mask_orig.astype(np.uint8), (w_feat, h_feat), interpolation=cv2.INTER_NEAREST)
        masks_down.append(torch.from_numpy(mask_resized).bool().to(device))
        
        x, y, w, h = m['bbox']
        x1 = x * scale_w
        y1 = y * scale_h
        x2 = (x + w) * scale_w
        y2 = (y + h) * scale_h
        boxes_feat.append([x1, y1, x2, y2])
        
    return S_feat, boxes_feat, masks_down

def generate_single_sam_frame(img_path_a, img_path_b, region_head, frame_extractor, device, target_size=(256, 256)):
    """
    核心函数：输入两张图片路径，返回式(14) 所需的两份原料，形状 [2, 3, H, W]
    （第 0 份为区域项 F_reg，第 1 份为背景项 S_proj）。

    α 的混合与式(15) 的 1x1 卷积不在这里做 —— 它们是训练期的可学习部分
    （见 model/trainer.py 的 SAMFrameFusion），缓存只保留混合前的原料。
    """
    # 1. 读取并预处理图像
    img_a = cv2.imread(img_path_a)
    img_b = cv2.imread(img_path_b)
    
    if img_a is None or img_b is None:
        raise FileNotFoundError(f"Image not found: {img_path_a} or {img_path_b}")

    img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
    
    # 调整大小以匹配模型输入
    img_a = cv2.resize(img_a, target_size)
    img_b = cv2.resize(img_b, target_size)
    orig_h, orig_w = target_size

    # 2. SAM 编码 Image A
    predictor_a = SamPredictor(sam)
    predictor_a.set_image(img_a)
    S1 = predictor_a.features.squeeze(0) # [256, 64, 64]
    masks_data_1 = mask_generator.generate(img_a)
    valid_masks_1 = filter_masks(masks_data_1)
    S1_feat, boxes_1, masks_1_down = process_masks(S1, valid_masks_1, orig_h, orig_w, device)

    # 3. SAM 编码 Image B
    predictor_b = SamPredictor(sam)
    predictor_b.set_image(img_b)
    S2 = predictor_b.features.squeeze(0)
    masks_data_2 = mask_generator.generate(img_b)
    valid_masks_2 = filter_masks(masks_data_2)
    S2_feat, boxes_2, masks_2_down = process_masks(S2, valid_masks_2, orig_h, orig_w, device)

    # 4. 提取区域特征
    desc1, kpts1 = extract_region_features(S1_feat, boxes_1, region_head)
    desc2, kpts2 = extract_region_features(S2_feat, boxes_2, region_head)

    # 5. 匹配并找到未匹配区域
    unmatched_idx1, unmatched_idx2, _, _ = find_unmatched_regions(
        desc1, kpts1, desc2, kpts2, matcher, threshold=0.2
    )

    U1_masks = [masks_1_down[i] for i in unmatched_idx1] if len(unmatched_idx1) > 0 else []
    U2_masks = [masks_2_down[i] for i in unmatched_idx2] if len(unmatched_idx2) > 0 else []
    U1_feats = desc1[unmatched_idx1] if len(unmatched_idx1) > 0 else torch.empty(0, 256, device=device)
    U2_feats = desc2[unmatched_idx2] if len(unmatched_idx2) > 0 else torch.empty(0, 256, device=device)

    # 6. 构建区域项与背景项
    F_reg, S_proj = frame_extractor(S1_feat, S2_feat, U1_masks, U2_masks, U1_feats, U2_feats)

    return torch.stack([F_reg, S_proj], dim=0).cpu().half()


def process_dataset_split(root_dir, split, region_head, frame_extractor, device, target_size=(256, 256), save_root_dir='./saved_sam_frames'):
    """
    处理单个划分（train/val/test）的数据集
    适配：A/B文件夹下直接存放图片，无场景子目录
    """
    # 构建路径
    img_dir_a = os.path.join(root_dir, split, 'A')
    img_dir_b = os.path.join(root_dir, split, 'B')
    save_root = os.path.join(save_root_dir, split)
    
    # 创建保存目录
    os.makedirs(save_root, exist_ok=True)

    # 直接获取 A 文件夹下的所有图片（无场景子文件夹）
    filenames = [f for f in os.listdir(img_dir_a) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(filenames) == 0:
        print(f"Warning: No images found in {img_dir_a}")
        return

    # 遍历所有图片对
    for filename in tqdm(filenames, desc=f"Processing {split}"):
        save_file = os.path.join(save_root, filename.replace('.png', '.pt').replace('.jpg', '.pt').replace('.jpeg', '.pt'))
        
        # 跳过已处理的文件
        if os.path.exists(save_file):
            continue
            
        path_a = os.path.join(img_dir_a, filename)
        path_b = os.path.join(img_dir_b, filename)
        
        try:
            with torch.no_grad():
                sam_frames = generate_single_sam_frame(path_a, path_b, region_head, frame_extractor, device, target_size)

            torch.save(sam_frames, save_file)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description="Offline SAM Frame Preprocessing for Levir-MCI")
    parser.add_argument('--root_dir', type=str, default='./Levir-MCI-dataset/images', help="Path to Levir-MCI images directory")
    parser.add_argument('--target_size', type=tuple, default=(256, 256), help="Input image size")
    parser.add_argument('--save_dir', type=str, default='./saved_sam_frames',
                        help="输出的缓存目录，需与 train_sam.py 的 --sam_folder 一致")
    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化模块
    print("Initializing SAM modules...")
    os.makedirs(args.save_dir, exist_ok=True)
    weights_path = os.path.join(args.save_dir, 'sam_frame_extractor.pth')

    region_head = RegionHead(in_channels=256, hidden=128, out_dim=256).to(device)
    frame_extractor = SAMFrameExtractor(d=256, C=3, H=256, W=256).to(device)

    # 提取器（含区域描述子的随机投影与背景投影）的权重必须和缓存配套：
    # 已经有缓存时直接复用旧权重，保证再次运行不会用新的随机初始化覆盖语义；
    # 首次运行则把这次的随机初始化存下来，供 predict.py 在线生成时复用。
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device)
        region_head.load_state_dict(ckpt['region_head'])
        frame_extractor.load_state_dict(ckpt['frame_extractor'])
        print(f"复用已有提取器权重：{weights_path}")
    else:
        torch.save({'region_head': region_head.state_dict(),
                    'frame_extractor': frame_extractor.state_dict()}, weights_path)
        print(f"已保存提取器权重：{weights_path}")

    sam.to(device)
    matcher.to(device)

    # 依次处理 train, val, test 三个划分
    splits = ['train', 'val', 'test']
    for split in splits:
        print(f"\n===== Processing {split.upper()} Split =====")
        process_dataset_split(
            root_dir=args.root_dir,
            split=split,
            region_head=region_head,
            frame_extractor=frame_extractor,
            device=device,
            target_size=args.target_size,
            save_root_dir=args.save_dir
        )

    print("\nAll splits processed successfully!")
    print("注意：缓存的每个 .pt 现在是 [2, 3, H, W]（区域项 + 背景项），")
    print("      α 混合与式(15) 的 1x1 卷积在训练时由 SAMFrameFusion 完成。")

if __name__ == '__main__':
    main()