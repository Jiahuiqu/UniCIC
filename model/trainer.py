# Copyright (c) Duowang Zhu.
# All rights reserved.

from functools import partial
import math
import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union, Callable, Any

import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange, repeat
from torch.utils.checkpoint import checkpoint

try:
    from .x3d import create_x3d
except ImportError:
    from model.x3d import create_x3d

# from model.change_decoder import ChangeDecoder
# from model.caption_decoder import CaptionDecoder
# from model.utils import weight_init


class Encoder(nn.Module):
    """
    Encoder model based on X3D architecture with feature enhancement capabilities.
    
    This encoder processes video frames using X3D architecture and enhances
    intermediate features using learnable tokens and temporal differences.
    """
    
    def __init__(self, args: Any, embed_dims: List[int]) -> None:
        """
        Initialize the encoder.
        
        Args:
            args: Configuration arguments
            embed_dims: Dimensions of embeddings at each stage
        """
        super().__init__()
        self.args = args
        self.embed_dims = embed_dims  # [24, 24, 48, 96]
        c1, c2, c3, c4 = embed_dims

        # Initialize X3D backbone
        self.x3d = create_x3d(input_clip_length=4, depth_factor=5.0)

        # Load pretrained weights
        try:
            state_dict = torch.load(args.pretrained, map_location='cpu')['model_state']
            msg = self.x3d.load_state_dict(state_dict, strict=True)
            print(f'Load pretrained weight: {args.pretrained}, {msg}.')
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")

        # Learnable perception frames for change extrcation
        self.perception_frames = nn.Parameter(
            torch.randn(1, 3, 2, args.in_height, args.in_width), 
            requires_grad=True
        )

        # Feature enhancement layers for each stage
        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    dim, 
                    dim, 
                    kernel_size=1, 
                    stride=1, 
                    padding=0, 
                    bias=False
                ),
                nn.ReLU()
            ) for dim in embed_dims
        ])

        self.up_c4 = nn.Sequential(
            nn.Conv2d(c4, c3, kernel_size=1, bias=False),
            nn.ConvTranspose2d(c3, c3, kernel_size=4, stride=2, padding=1)
        )
        self.up_c3 = nn.Sequential(
            nn.Conv2d(c3, c2, kernel_size=1, bias=False),
            nn.ConvTranspose2d(c2, c2, kernel_size=4, stride=2, padding=1)
        )
        self.up_c2 = nn.Sequential(
            nn.Conv2d(c2, c1, kernel_size=1, bias=False),
            nn.ConvTranspose2d(c1, c1, kernel_size=4, stride=2, padding=1)
        )

    def enhance(self, x: torch.Tensor, fc: nn.Module) -> torch.Tensor:
        """
        Enhance the binary change frame using temporal information from pre/post frames.
        
        Args:
            x: Input tensor with shape [B, C, T, H, W] where T is the temporal dimension
            fc: Feature enhancement module for refining temporal features

        Returns:
            torch.Tensor: Enhanced tensor with same shape as input, with middle
                        frame enhanced using temporal information

        Note:
            The enhancement is only applied to the middle frame (T//2), leaving
            other frames unchanged.
        """
        
        # Get the pre and post frames based on configuration
        pre_frame = x[:, :, 0]  # First frame
        post_frame = x[:, :, 3]  # Specified post frame
        
        # Calculate temporal difference features
        temporal_diff = torch.abs(pre_frame - post_frame)
        
        # Apply feature enhancement module to extract refinement features
        enhancement_features = fc(temporal_diff)
        
        # Apply enhancement to middle frame only (residual enhancement)
        enhanced_x = x.clone()
        enhanced_x[:, :, 1] = x[:, :, 1] + enhancement_features  # ← 感知帧 1
        
        return enhanced_x

    def base_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_feat = x.clone()
        c1 = c2 = c3 = c4 = None
        for i in range(5):
            x_feat = checkpoint(self.x3d.blocks[i], x_feat)
        
            if i < 4:
                x_feat = self.enhance(x_feat, self.fc[i])
                feat = x_feat[:, :, 1]
                if i == 0:
                    c1 = feat
                elif i == 1:
                    c2 = feat
                elif i == 2:
                    c3 = feat
                elif i == 3:
                    c4 = feat

        c3f = c3 + self.up_c4(c4)
        c2f = c2 + self.up_c3(c3f)
        c1f = c1 + self.up_c2(c2f)
        cd_feature = c1f

        cc_feature = x_feat[:, :, 2]

        return cd_feature, cc_feature

     

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with input and target frames.
        
        Args:
            x: Input frame tensor with shape [B, C, H, W]
            y: Target frame tensor with shape [B, C, H, W]
            
        Returns:
            List of feature tensors from different stages
        """        
        # Expand tokens to match batch size
        expand_percep_frames = repeat(self.perception_frames, '1 c t h w -> b c t h w', b=x.shape[0])

        # Combine into 3-frame sequence [input, token, target]
        frames = torch.cat([
            x.unsqueeze(2),
            expand_percep_frames,
            y.unsqueeze(2)
        ], dim=2)

        # Process through network
        cd_feature, cc_feature = self.base_forward(frames)

        return cd_feature, cc_feature
    
class Trainer(nn.Module):
    def __init__(self, args: Any) -> None:
        super().__init__()
        self.args = args
        self.embed_dims = [24, 24, 48, 96]
        self.encoder = Encoder(args, self.embed_dims)

    def forward(self, x, y):
        cd_feature, cc_feature = self.encoder(x, y)
        return cd_feature, cc_feature
    

class SAMFrameFusion(nn.Module):
    """把离线缓存的区域项与背景项合成 SAM 感知帧（论文式(14)(15)）。

        F_SAM = alpha * F_reg + (1 - alpha) * S_proj    (14)
        I_SAM = Conv1x1(F_SAM)                          (15)

    alpha 为可学习标量，与论文式(14) 里 "a learnable fusion weight" 一致。
    原先 α 的混合发生在离线预处理阶段（感知帧一次性算好后缓存成 .pt），α 拿不到
    任何梯度；这里把混合挪进训练计算图，缓存只保留混合前的两份原料。

    两个 1x1 卷积不设 bias：当缓存缺失（两份原料全零）时输出恒为零，感知帧退化
    为纯随机的可学习基帧，与「不使用 SAM 先验」的行为一致。
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.channel_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, f_reg: torch.Tensor, s_proj: torch.Tensor) -> torch.Tensor:
        fused = self.alpha * f_reg + (1.0 - self.alpha) * s_proj
        return self.channel_conv(self.final_conv(fused))


class SAMEncoder(nn.Module):
    """
    Encoder model based on X3D architecture with SAM frame integration.

    This encoder processes video frames using X3D architecture.
    The perception sequence is constructed as: [Input, Learnable_Frame_1, Learnable_Base_2 + SAM_Frame, Target]
    """
    
    def __init__(self, args: any, embed_dims: list) -> None:
        """
        Initialize the encoder.
        
        Args:
            args: Configuration arguments
            embed_dims: Dimensions of embeddings at each stage
        """
        super().__init__()
        self.args = args
        self.embed_dims = embed_dims  # [24, 24, 48, 96]
        c1, c2, c3, c4 = embed_dims

        # Initialize X3D backbone
        self.x3d = create_x3d(input_clip_length=4, depth_factor=5.0)

        # Load pretrained weights
        try:
            # 注意：这里假设 args.pretrained 存在，如果为 None 需处理
            if hasattr(args, 'pretrained') and args.pretrained:
                state_dict = torch.load(args.pretrained, map_location='cpu')['model_state']
                msg = self.x3d.load_state_dict(state_dict, strict=True)
                print(f'Load pretrained weight: {args.pretrained}, {msg}.')
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")

        # 1. Learnable perception frame for the first slot (Index 1 in the 4-frame sequence)
        # Shape: [1, C, H, W]
        self.perception_frame_1 = nn.Parameter(
            torch.randn(1, 3, args.in_height, args.in_width), 
            requires_grad=True
        )

        # 2. Learnable BASE for the second slot (Index 2 in the 4-frame sequence)
        # This will be added to the input SAM frame.
        # Shape: [1, C, H, W]
        self.perception_frame_2_base = nn.Parameter(
            torch.randn(1, 3, args.in_height, args.in_width),
            requires_grad=True
        )

        # 3. SAM 感知帧的融合模块：把缓存的区域项/背景项合成 I_SAM^P（式(14)(15)）
        self.frame_fusion = SAMFrameFusion(in_channels=3, out_channels=3)

        # Feature enhancement layers for each stage
        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    dim, 
                    dim, 
                    kernel_size=1, 
                    stride=1, 
                    padding=0, 
                    bias=False
                ),
                nn.ReLU()
            ) for dim in embed_dims
        ])

        self.up_c4 = nn.Sequential(
            nn.Conv2d(c4, c3, kernel_size=1, bias=False),
            nn.ConvTranspose2d(c3, c3, kernel_size=4, stride=2, padding=1)
        )
        self.up_c3 = nn.Sequential(
            nn.Conv2d(c3, c2, kernel_size=1, bias=False),
            nn.ConvTranspose2d(c2, c2, kernel_size=4, stride=2, padding=1)
        )
        self.up_c2 = nn.Sequential(
            nn.Conv2d(c2, c1, kernel_size=1, bias=False),
            nn.ConvTranspose2d(c1, c1, kernel_size=4, stride=2, padding=1)
        )

    def enhance(self, x: torch.Tensor, fc: nn.Module) -> torch.Tensor:
        """
        Enhance the binary change frame using temporal information from pre/post frames.
        """
        # Get the pre and post frames based on configuration
        pre_frame = x[:, :, 0]  # First frame (Input A)
        post_frame = x[:, :, 3]  # Last frame (Input B)
        
        # Calculate temporal difference features
        temporal_diff = torch.abs(pre_frame - post_frame)
        
        # Apply feature enhancement module to extract refinement features
        enhancement_features = fc(temporal_diff)
        
        # Apply enhancement to middle frame only (residual enhancement)
        # Note: In our new 4-frame sequence [A, P1, P2+SAM, B], 
        # the "middle" frames are indices 1 and 2. 
        # The original code enhanced index 1. 
        # If you want to enhance the combined perception frame (index 2) or both, logic might need adjustment.
        # However, sticking to "rest unchanged", we keep enhancing index 1 as per original logic, 
        # OR we might want to enhance the perception frames. 
        # Original code: enhanced_x[:, :, 1] = x[:, :, 1] + enhancement_features
        # Let's keep it exactly as original to minimize side effects, assuming index 1 is the primary perception target.
        enhanced_x = x.clone()
        enhanced_x[:, :, 1] = x[:, :, 1] + enhancement_features 
        
        return enhanced_x

    def base_forward(self, x: torch.Tensor):
        x_feat = x.clone()
        c1 = c2 = c3 = c4 = None
        for i in range(5):
            x_feat = checkpoint(self.x3d.blocks[i], x_feat)
        
            if i < 4:
                x_feat = self.enhance(x_feat, self.fc[i])
                feat = x_feat[:, :, 1]
                if i == 0:
                    c1 = feat
                elif i == 1:
                    c2 = feat
                elif i == 2:
                    c3 = feat
                elif i == 3:
                    c4 = feat

        c3f = c3 + self.up_c4(c4)
        c2f = c2 + self.up_c3(c3f)
        c1f = c1 + self.up_c2(c2f)
        cd_feature = c1f

        cc_feature = x_feat[:, :, 2]

        return cd_feature, cc_feature

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                sam_region: torch.Tensor = None, sam_bg: torch.Tensor = None) -> tuple:
        """
        Forward pass with input, target, and the cached SAM frame ingredients.

        Args:
            x: Input frame tensor (A) with shape [B, C, H, W]
            y: Target frame tensor (B) with shape [B, C, H, W]
            sam_region: 式(14) 区域项 [B, C, H, W]，由 preprocess_sam_frames.py 缓存
            sam_bg: 式(14) 背景项 [B, C, H, W]，同上。两者都为 None 时退化为无 SAM 先验。

        Returns:
            cd_feature, cc_feature
        """
        b, c, h, w = x.shape

        # 1. Prepare Perception Frame 1 (Learnable)
        # Expand to batch size
        p1 = repeat(self.perception_frame_1, '1 c h w -> b c h w', b=b)

        # 2. Prepare Perception Frame 2 (Learnable Base + SAM)
        # Expand base to batch size
        p2_base = repeat(self.perception_frame_2_base, '1 c h w -> b c h w', b=b)

        if sam_region is not None and sam_bg is not None:
            # α 在这里参与梯度回传（式(14)），再叠加可学习基帧
            p2 = self.frame_fusion(sam_region, sam_bg) + p2_base
        else:
            # Fallback if no SAM frame provided
            p2 = p2_base

        # 3. Construct the 4-frame sequence
        # Sequence: [Input(A), Perception_1, Perception_2(SAM+Base), Target(B)]
        # Dimensions: [B, C, 4, H, W]
        frames = torch.stack([
            x,      # Index 0: Time T-1 (Pre-change)
            p1,     # Index 1: Learnable Perception Frame 1
            p2,     # Index 2: Learnable Perception Frame 2 + SAM
            y       # Index 3: Time T+1 (Post-change)
        ], dim=2)

        # Process through network
        cd_feature, cc_feature = self.base_forward(frames)

        return cd_feature, cc_feature


class SAMTrainer(nn.Module):
    def __init__(self, args: any) -> None:
        super().__init__()
        self.args = args
        self.embed_dims = [24, 24, 48, 96]
        # Use the new SAMEncoder
        self.encoder = SAMEncoder(args, self.embed_dims)

    def forward(self, x, y, sam_region=None, sam_bg=None):
        """
        Args:
            x: Image A [B, C, H, W]
            y: Image B [B, C, H, W]
            sam_region: 式(14) 区域项 [B, C, H, W]
            sam_bg: 式(14) 背景项 [B, C, H, W]
        """
        cd_feature, cc_feature = self.encoder(x, y, sam_region, sam_bg)
        return cd_feature, cc_feature
    


# class SAMEncoderV1(nn.Module):
#     """
#     Encoder model based on X3D architecture with SAM frame integration.
    
#     This encoder processes video frames using X3D architecture.
#     The perception sequence is constructed as: [Input, Learnable_Frame_1, Learnable_Base_2 + SAM_Frame, Target]
#     """
    
#     def __init__(self, args: any, embed_dims: list) -> None:
#         """
#         Initialize the encoder.
        
#         Args:
#             args: Configuration arguments
#             embed_dims: Dimensions of embeddings at each stage
#         """
#         super().__init__()
#         self.args = args
#         self.embed_dims = embed_dims  # [24, 24, 48, 96]
#         c1, c2, c3, c4 = embed_dims

#         # Initialize X3D backbone
#         self.x3d = create_x3d(input_clip_length=4, depth_factor=5.0)

#         # Load pretrained weights
#         try:
#             # 注意：这里假设 args.pretrained 存在，如果为 None 需处理
#             if hasattr(args, 'pretrained') and args.pretrained:
#                 state_dict = torch.load(args.pretrained, map_location='cpu')['model_state']
#                 msg = self.x3d.load_state_dict(state_dict, strict=True)
#                 print(f'Load pretrained weight: {args.pretrained}, {msg}.')
#         except Exception as e:
#             print(f"Failed to load pretrained weights: {e}")

#         # 1. Learnable perception frame for the first slot (Index 1 in the 4-frame sequence)
#         # Shape: [1, C, H, W]
#         self.perception_frame_1 = nn.Parameter(
#             torch.randn(1, 3, args.in_height, args.in_width), 
#             requires_grad=True
#         )

#         # 2. Learnable BASE for the second slot (Index 2 in the 4-frame sequence)
#         # This will be added to the input SAM frame.
#         # Shape: [1, C, H, W]
#         self.perception_frame_2_base = nn.Parameter(
#             torch.randn(1, 3, args.in_height, args.in_width), 
#             requires_grad=True
#         )

#         # Feature enhancement layers for each stage
#         self.fc = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(
#                     dim, 
#                     dim, 
#                     kernel_size=1, 
#                     stride=1, 
#                     padding=0, 
#                     bias=False
#                 ),
#                 nn.ReLU()
#             ) for dim in embed_dims
#         ])

#         self.up_c4 = nn.Sequential(
#             nn.Conv2d(c4, c3, kernel_size=1, bias=False),
#             nn.ConvTranspose2d(c3, c3, kernel_size=4, stride=2, padding=1)
#         )
#         self.up_c3 = nn.Sequential(
#             nn.Conv2d(c3, c2, kernel_size=1, bias=False),
#             nn.ConvTranspose2d(c2, c2, kernel_size=4, stride=2, padding=1)
#         )
#         self.up_c2 = nn.Sequential(
#             nn.Conv2d(c2, c1, kernel_size=1, bias=False),
#             nn.ConvTranspose2d(c1, c1, kernel_size=4, stride=2, padding=1)
#         )

#     def enhance(self, x: torch.Tensor, fc: nn.Module) -> torch.Tensor:
#         """
#         Enhance the binary change frame using temporal information from pre/post frames.
#         """
#         # Get the pre and post frames based on configuration
#         pre_frame = x[:, :, 0]  # First frame (Input A)
#         post_frame = x[:, :, 3]  # Last frame (Input B)
        
#         # Calculate temporal difference features
#         temporal_diff = torch.abs(pre_frame - post_frame)
        
#         # Apply feature enhancement module to extract refinement features
#         enhancement_features = fc(temporal_diff)
        
#         # Apply enhancement to middle frame only (residual enhancement)
#         # Note: In our new 4-frame sequence [A, P1, P2+SAM, B], 
#         # the "middle" frames are indices 1 and 2. 
#         # The original code enhanced index 1. 
#         # If you want to enhance the combined perception frame (index 2) or both, logic might need adjustment.
#         # However, sticking to "rest unchanged", we keep enhancing index 1 as per original logic, 
#         # OR we might want to enhance the perception frames. 
#         # Original code: enhanced_x[:, :, 1] = x[:, :, 1] + enhancement_features
#         # Let's keep it exactly as original to minimize side effects, assuming index 1 is the primary perception target.
#         enhanced_x = x.clone()
#         enhanced_x[:, :, 1] = x[:, :, 1] + enhancement_features 
        
#         return enhanced_x

#     def base_forward(self, x: torch.Tensor):
#         x_feat = x.clone()
#         c1 = c2 = c3 = c4 = None
#         for i in range(5):
#             x_feat = checkpoint(self.x3d.blocks[i], x_feat)
        
#             if i < 4:
#                 x_feat = self.enhance(x_feat, self.fc[i])
#                 feat = x_feat[:, :, 1]
#                 if i == 0:
#                     c1 = feat
#                 elif i == 1:
#                     c2 = feat
#                 elif i == 2:
#                     c3 = feat
#                 elif i == 3:
#                     c4 = feat

#         c3f = c3 + self.up_c4(c4)
#         c2f = c2 + self.up_c3(c3f)
#         c1f = c1 + self.up_c2(c2f)
#         cd_feature = c1f

#         cc_feature = x_feat[:, :, 2]

#         return cd_feature, cc_feature

#     def forward(self, x: torch.Tensor, y: torch.Tensor, sam_frame: torch.Tensor = None) -> tuple:
#         """
#         Forward pass with input, target, and optional SAM frame.
        
#         Args:
#             x: Input frame tensor (A) with shape [B, C, H, W]
#             y: Target frame tensor (B) with shape [B, C, H, W]
#             sam_frame: SAM frame tensor with shape [B, C, H, W]. 
#                        If None, zeros are used.
            
#         Returns:
#             cd_feature, cc_feature
#         """        
#         b, c, h, w = x.shape

#         # 1. Prepare Perception Frame 1 (Learnable)
#         # Expand to batch size
#         p1 = repeat(self.perception_frame_1, '1 c h w -> b c h w', b=b)

#         # 2. Prepare Perception Frame 2 (Learnable Base + SAM)
#         # Expand base to batch size
#         p2_base = repeat(self.perception_frame_2_base, '1 c h w -> b c h w', b=b)
        
#         if sam_frame is not None:
#             # Ensure sam_frame is on the same device and has correct shape
#             if sam_frame.shape != (b, c, h, w):
#                 # Handle potential size mismatch if necessary, e.g., interpolate
#                 # For now, assume dataset returns correct size (256x256 as per dataset code)
#                 pass
#             p1 = sam_frame + p1
#             p2 = sam_frame +  p2_base
            
#         else:
#             # Fallback if no SAM frame provided
#             p1 = p1
#             p2 = p2_base

#         # 3. Construct the 4-frame sequence
#         # Sequence: [Input(A), Perception_1, Perception_2(SAM+Base), Target(B)]
#         # Dimensions: [B, C, 4, H, W]
#         frames = torch.stack([
#             x,
#             p1,    
#             p2,
#             y
#         ], dim=2)

#         # Process through network
#         cd_feature, cc_feature = self.base_forward(frames)

#         return cd_feature, cc_feature


# class SAMTrainer(nn.Module):
#     def __init__(self, args: any) -> None:
#         super().__init__()
#         self.args = args
#         self.embed_dims = [24, 24, 48, 96]
#         # Use the new SAMEncoder
#         self.encoder = SAMEncoderV1(args, self.embed_dims)

#     def forward(self, x, y, sam_frame=None):
#         """
#         Args:
#             x: Image A [B, C, H, W]
#             y: Image B [B, C, H, W]
#             sam_frame: SAM embedding/frame [B, C, H, W]
#         """
#         cd_feature, cc_feature = self.encoder(x, y, sam_frame)
#         return cd_feature, cc_feature
    

if __name__ == '__main__':
    import argparse
    args = argparse.Namespace(
        pretrained=None,       
        in_height=256,
        in_width=256,
        num_perception_frame=2
    )

    model = Encoder(args, embed_dims=[24, 24, 48, 96])
    model.eval()  
    x = torch.randn(1, 3, 256, 256) 
    y = torch.randn(1, 3, 256, 256) 

    with torch.no_grad():
        cd_feature , cc_feature = model(x, y)

    print('变化检测融合特征 shape:', cd_feature.shape)
    print('变化描述最终特征 shape:', cc_feature.shape)


# class Trainer(nn.Module):
#     """
#     Complete model with encoder and decoder for video frame enhancement.
#     """
    
#     def __init__(self, args: Any) -> None:
#         """
#         Initialize the trainer with encoder and decoder.
        
#         Args:
#             args: Configuration arguments
#         """
#         super().__init__()
#         self.args = args
    
#         # Define embedding dimensions for each stage
#         self.embed_dims = [24, 24, 48, 96]
        
#         # Initialize encoder and decoder
#         self.encoder = Encoder(args, self.embed_dims)
        
#         # For binary change detection and change caption task
#         if args.num_perception_frame == 1 and 'CD' in args.dataset:
#             self.decoder = ChangeDecoder(args, in_dim=self.embed_dims, has_sigmoid=True)
#             # Initialize decoder weights
#             weight_init(self.decoder)

#         # For semantic change detection task
#         elif args.num_perception_frame == 3:
#             self.decoder_pre = ChangeDecoder(args, in_dim=self.embed_dims)
#             self.decoder_post = ChangeDecoder(args, in_dim=self.embed_dims)
#             self.decoder_change = ChangeDecoder(args, in_dim=self.embed_dims, has_sigmoid=True)
#             # Initialize decoder weights
#             weight_init(self.decoder_pre)
#             weight_init(self.decoder_post)
#             weight_init(self.decoder_change)

#         # For building damage assessment task
#         elif args.num_perception_frame == 2:
#             self.decoder_cls = ChangeDecoder(args, in_dim=self.embed_dims)
#             self.decoder_loc = ChangeDecoder(args, in_dim=self.embed_dims, has_sigmoid=True)
#             # Initialize decoder weights
#             weight_init(self.decoder_cls)
#             weight_init(self.decoder_loc)

#         # For change caption task
#         elif args.num_perception_frame == 1 and 'CC' in args.dataset:
#             self.decoder = CaptionDecoder(args)
#         else:
#             assert False

#     def update_bcd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass through the complete model.

#         Args:
#             x: Input frame tensor with shape [B, C, H, W]
#             y: Target frame tensor with shape [B, C, H, W]

#         Returns:
#             Predicted frame tensor
#         """
#         # Extract features using encoder
#         features = self.encoder(x, y)

#         # perception feature
#         perception_change_feat = list(map(lambda x: x[0], features))

#         # Generate prediction using decoder
#         prediction = self.decoder(perception_change_feat)

#         return prediction
    
#     def update_scd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass through the complete model.

#         Args:
#             x: Input frame tensor with shape [B, C, H, W]
#             y: Target frame tensor with shape [B, C, H, W]

#         Returns:
#             Predicted frame tensor
#         """
#         # Extract features using encoder
#         features = self.encoder(x, y)
        
#         # Generate prediction using decoder
#         perception_pre_feat = list(map(lambda x: x[0], features))
#         perception_change_feat = list(map(lambda x: x[1], features))
#         perception_post_feat = list(map(lambda x: x[2], features))

#         pre_mask = self.decoder_pre(perception_pre_feat)
#         post_mask = self.decoder_post(perception_post_feat)
#         change_mask = self.decoder_change(perception_change_feat)

#         return pre_mask, post_mask, change_mask
    
#     def update_bda(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass through the complete model.

#         Args:
#             x: Input frame tensor with shape [B, C, H, W]
#             y: Target frame tensor with shape [B, C, H, W]

#         Returns:
#             Predicted frame tensor
#         """
#         # Extract features using encoder
#         features = self.encoder(x, y)

#         # perception feature
#         perception_cls_feat = list(map(lambda x: x[0], features))
#         perception_loc_feat = list(map(lambda x: x[1], features))

#         # Generate prediction using decoder
#         pred_cls = self.decoder_cls(perception_cls_feat)
#         pred_loc = self.decoder_loc(perception_loc_feat)

#         return pred_cls, pred_loc
    
#     def update_cc(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass through the complete model.

#         Args:
#             x: Input frame tensor with shape [B, C, H, W]
#             y: Target frame tensor with shape [B, C, H, W]

#         Returns:
#             Predicted frame tensor
#         """
#         # Extract features using encoder
#         features = self.encoder(x, y, output_final=True)

#         return features
