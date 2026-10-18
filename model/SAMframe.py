import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision
import os

from torchvision.ops import roi_align
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from lightglue import LightGlue

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SAM 模型
sam_checkpoint = "/media/xd132/USER/CXT/Change-Agent-main_backup/Multi_change/sam_vit_h_4b8939.pth" 
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
print(f"✓ SAM 模型加载成功：{sam_checkpoint}")

# LightGlue 匹配器
matcher = LightGlue(features='superpoint').eval().to(device)
print("✓ LightGlue 模型加载成功")


def process_image_sam(image_path, mask_generator, sam_model, device, target_size=(256, 256)):
    """
    输入图像路径，返回：
        S: SAM 特征图 (C_s, h_feat, w_feat)  C_s=256, h_feat=w_feat=64
        masks_down: 筛选后的二值掩码列表，每个 (h_feat, w_feat)
        boxes_feat: 边界框列表 (特征图坐标系)，每个 [x1,y1,x2,y2]
        valid_masks_orig: 原始掩码数据（用于调试）
    """
    # 读取并预处理图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    orig_h, orig_w = target_size

    # SAM 编码 - 使用 SamPredictor 统一管理
    predictor = SamPredictor(sam_model)
    predictor.set_image(img)
    S = predictor.features  # (1, C_s, h_feat, w_feat)
    _, C_s, h_feat, w_feat = S.shape
    print(f"特征图尺寸：{h_feat} x {w_feat}, 通道数：{C_s}")

    # 生成自动掩码
    masks_data = mask_generator.generate(img)
    print(f"原始掩码数量：{len(masks_data)}")

    # 筛选
    min_area = 100
    iou_thresh = 0.8
    valid_masks = [m for m in masks_data if m['predicted_iou'] > iou_thresh and m['area'] > min_area]

    # NMS
    if len(valid_masks) > 0:
        boxes_orig = [m['bbox'] for m in valid_masks]  # [x,y,w,h]
        scores = [m['predicted_iou'] for m in valid_masks]
        boxes_xyxy = []
        for b in boxes_orig:
            x, y, w, h = b
            boxes_xyxy.append([x, y, x + w, y + h])
        boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        keep = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
        valid_masks = [valid_masks[i] for i in keep]

    print(f"筛选后掩码数量：{len(valid_masks)}")

    # 缩放因子
    scale_h = h_feat / orig_h
    scale_w = w_feat / orig_w

    masks_down = []
    boxes_feat = []
    for m in valid_masks:
        # 掩码下采样
        mask_orig = m['segmentation']
        mask_resized = cv2.resize(mask_orig.astype(np.uint8), (w_feat, h_feat), interpolation=cv2.INTER_NEAREST)
        masks_down.append(torch.from_numpy(mask_resized).bool().to(device))

        # 边界框转换
        x, y, w, h = m['bbox']
        x1 = x * scale_w
        y1 = y * scale_h
        x2 = (x + w) * scale_w
        y2 = (y + h) * scale_h
        boxes_feat.append([x1, y1, x2, y2])

    S = S.squeeze(0)  # (C_s, h_feat, w_feat)
    return S, masks_down, boxes_feat, valid_masks


class RegionHead(nn.Module):
    """将 RoI 特征映射为 d 维描述符"""
    def __init__(self, in_channels=256, hidden=128, out_dim=256):
        """
        【修复】out_dim 必须为 256，与 LightGlue SuperPoint 特征维度匹配
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.fc = nn.Linear(hidden * 7 * 7, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.flatten(1)
        x = self.fc(x)
        return x


def extract_region_features(S, boxes, region_head, output_size=(14, 14), pool_size=(7, 7)):
    """
    S: (C_s, h, w)
    boxes: list of [x1,y1,x2,y2]
    返回:
        desc: (N, d)
        centers: (N, 2)
    """
    if len(boxes) == 0:
        return torch.empty(0, region_head.fc.out_features).to(S.device), torch.empty(0, 2).to(S.device)

    S_batch = S.unsqueeze(0)  # (1, C, h, w)
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=S.device)
    
    # 【修复】添加 batch index 列 (全 0)，形状变为 (N, 5)
    batch_idx = torch.zeros(len(boxes), 1, dtype=torch.float32, device=S.device)
    boxes_roi = torch.cat([batch_idx, boxes_tensor], dim=1)
    
    # 【修复】直接传入 boxes_roi Tensor，不要包列表
    roi_feats = roi_align(S_batch, boxes_roi, output_size=output_size, spatial_scale=1.0)
    
    pooled = F.adaptive_avg_pool2d(roi_feats, pool_size)
    desc = region_head(pooled)
    centers = (boxes_tensor[:, :2] + boxes_tensor[:, 2:]) / 2
    return desc, centers


def find_unmatched_regions(desc1, kpts1, desc2, kpts2, matcher, threshold=0.2):
    """
    使用 LightGlue 进行区域匹配，筛选未匹配区域
    【修复】输入形状改为 (B, N, D)，移除 transpose
    """
    if desc1.shape[0] == 0:
        return (torch.arange(0), 
                torch.arange(desc2.shape[0]) if desc2.shape[0] > 0 else torch.empty(0, dtype=torch.long))
    if desc2.shape[0] == 0:
        return (torch.arange(desc1.shape[0]), 
                torch.arange(0))

    # 【修复】LightGlue 期望输入形状为 (B, N, D)
    data = {
    "image0": {"keypoints": kpts1.unsqueeze(0), "descriptors": desc1.unsqueeze(0)},
    "image1": {"keypoints": kpts2.unsqueeze(0), "descriptors": desc2.unsqueeze(0)},
     }

    with torch.no_grad():
        pred = matcher(data)

    matches0 = pred['matches0'][0]
    scores0 = pred['matching_scores0'][0]
    matches1 = pred['matches1'][0]
    scores1 = pred['matching_scores1'][0]

    unmatched_idx1 = torch.where((matches0 == -1) | (scores0 < threshold))[0].cpu()
    unmatched_idx2 = torch.where((matches1 == -1) | (scores1 < threshold))[0].cpu()
    
    return unmatched_idx1, unmatched_idx2, scores0, scores1


class SAMFrameExtractor(nn.Module):
    """SAM 区域/背景原料提取器，只在离线预处理阶段使用，参数随机固定不再更新。

    输出的两份原料分别对应论文式(14) 加号两侧，但不做 α 混合，也不做式(15) 的
    1x1 卷积：

        F_reg  —— 区域项 Σ_{b∈U1∪U2} d_{t,b} / N_p，未匹配区域的 RoI 描述子
                  按掩码回填后逐像素平均（N_p 为覆盖每个空间位置的区域数）
        S_proj —— 背景项 F_proj((S1 + S2) / 2)

    α 与式(15) 的卷积改由训练期的 SAMFrameFusion 承担，这样 α 才真正参与反向
    传播（论文式(14) 明确称其为 learnable fusion weight）。
    """

    def __init__(self, d=256, C=3, H=256, W=256, sam_feature_channels=256):
        """
        d: 区域描述子维度，与 RegionHead 输出维度匹配
        C: 区域项/背景项的中间通道数
        """
        super().__init__()
        self.d = d
        self.C = C
        self.H = H
        self.W = W

        self.proj = nn.Linear(d, C)
        self.bg_proj = nn.Conv2d(sam_feature_channels, C, kernel_size=1)  # SAM 特征通道 256

    def forward(self, S1, S2, U1_masks, U2_masks, U1_feats, U2_feats):
        device = S1.device

        # 投影
        U1_proj = self.proj(U1_feats) if U1_feats.shape[0] > 0 else torch.empty(0, self.C, device=device)
        U2_proj = self.proj(U2_feats) if U2_feats.shape[0] > 0 else torch.empty(0, self.C, device=device)

        # 初始化空帧
        F_sam = torch.zeros(self.C, self.H, self.W, device=device)
        weight = torch.zeros(self.H, self.W, device=device)

        def upsample_mask(mask, size):
            mask_float = mask.float().unsqueeze(0).unsqueeze(0)
            up = F.interpolate(mask_float, size=size, mode='nearest')
            return up.squeeze(0).squeeze(0).bool()

        # 回填时相 1
        for mask, feat in zip(U1_masks, U1_proj):
            mask_up = upsample_mask(mask, (self.H, self.W))
            F_sam[:, mask_up] += feat.unsqueeze(1)
            weight[mask_up] += 1

        # 回填时相 2
        for mask, feat in zip(U2_masks, U2_proj):
            mask_up = upsample_mask(mask, (self.H, self.W))
            F_sam[:, mask_up] += feat.unsqueeze(1)
            weight[mask_up] += 1

        weight = weight.clamp(min=1)
        F_reg = F_sam / weight.unsqueeze(0)

        # 背景项
        S1_up = F.interpolate(S1.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=False).squeeze(0)
        S2_up = F.interpolate(S2.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=False).squeeze(0)
        S_proj = self.bg_proj(((S1_up + S2_up) / 2).unsqueeze(0)).squeeze(0)

        return F_reg, S_proj


def build_sam_frames(img_path1, img_path2,
                     mask_generator, sam_model,
                     region_head, matcher, frame_extractor,
                     device, d=256, C=3, H=256, W=256, match_thresh=0.2):
    """对一对图像生成式(14) 所需的区域项与背景项，返回 (F_reg, S_proj)。"""
    # 处理时相 1
    S1, masks1, boxes1, _ = process_image_sam(img_path1, mask_generator, sam_model, device, target_size=(H, W))
    # 处理时相 2
    S2, masks2, boxes2, _ = process_image_sam(img_path2, mask_generator, sam_model, device, target_size=(H, W))

    # 提取区域特征
    desc1, kpts1 = extract_region_features(S1, boxes1, region_head)
    desc2, kpts2 = extract_region_features(S2, boxes2, region_head)

    # 匹配
    unmatched_idx1, unmatched_idx2, _, _ = find_unmatched_regions(desc1, kpts1, desc2, kpts2, matcher, threshold=match_thresh)

    # 获取未匹配区域
    U1_masks = [masks1[i] for i in unmatched_idx1] if len(unmatched_idx1) > 0 else []
    U2_masks = [masks2[i] for i in unmatched_idx2] if len(unmatched_idx2) > 0 else []
    U1_feats = desc1[unmatched_idx1] if len(unmatched_idx1) > 0 else torch.empty(0, d, device=device)
    U2_feats = desc2[unmatched_idx2] if len(unmatched_idx2) > 0 else torch.empty(0, d, device=device)

    # 构建区域项与背景项
    F_reg, S_proj = frame_extractor(S1, S2, U1_masks, U2_masks, U1_feats, U2_feats)
    return F_reg, S_proj


# ==================== 测试 ====================
if __name__ == "__main__":
    # 初始化模块
    # 【修复】out_dim=256 与 LightGlue 匹配
    region_head = RegionHead(in_channels=256, hidden=128, out_dim=256).to(device)
    frame_extractor = SAMFrameExtractor(d=256, C=3, H=256, W=256).to(device)

    # 替换为你的图像路径
    img1 = "/media/xd132/USER/CXT/Change-Agent-main_backup/Multi_change/Levir-MCI-dataset/images/train/A/train_000001.png"
    img2 = "/media/xd132/USER/CXT/Change-Agent-main_backup/Multi_change/Levir-MCI-dataset/images/train/B/train_000001.png"

    save_dir = "/media/xd132/USER/CXT/Change-Agent-main_backup/results/"
    os.makedirs(save_dir, exist_ok=True)

    F_reg, S_proj = build_sam_frames(img1, img2,
                                     mask_generator, sam,
                                     region_head, matcher, frame_extractor,
                                     device, d=256, C=3, H=256, W=256, match_thresh=0.2)

    print("区域项维度:", F_reg.shape, "背景项维度:", S_proj.shape)  # 均应为 (3, 256, 256)

    def save_pseudo_color_frame(F_sam, save_path):
        # 压缩为单通道并归一化
        F_single = torch.mean(F_sam, dim=0).detach().cpu().numpy()
        F_single = (F_single - F_single.min()) / (F_single.max() - F_single.min() + 1e-8)
        F_single = (F_single * 255).astype(np.uint8)
        # 伪彩色映射
        pseudo_img = cv2.applyColorMap(F_single, cv2.COLORMAP_JET)
        # 保存
        cv2.imwrite(save_path, pseudo_img)
        print(f"✅ 伪彩色感知帧已保存至：{save_path}")

    # 执行保存
    save_pseudo_color_frame(F_reg, os.path.join(save_dir, "perception_frame_region_pseudo_color.png"))
    save_pseudo_color_frame(S_proj, os.path.join(save_dir, "perception_frame_background_pseudo_color.png"))