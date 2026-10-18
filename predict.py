import os
import os.path

import argparse
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure

from model.model_encoder_att import SAMEncoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer
from utils_tool.utils import *
from imageio.v2 import imread


# compute_change_map(path_A, path_B)函数: 生成一个掩膜mask用来表示两个图像之间的变化区域
'''
Args:
    path_A: 图像A的路径
    path_B: 图像B的路径
Returns:
    change_map: 变化区域的掩膜
'''
# compute_change_caption(path_A, path_B)函数：生成一个文本用于描述两个图像之间变化
'''
Args:
    path_A: 图像A的路径
    path_B: 图像B的路径
Returns:
    caption: 变化描述文本
'''
class Change_Perception(object):
    def define_args(self):

        parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Interpretation')

        # data
        parser.add_argument('--data_folder', default='./Levir-MCI-dataset/images',
                            help='folder with data files')
        parser.add_argument('--list_path', default='./data/LEVIR_MCI/',
                            help='path of the data lists')
        parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
        parser.add_argument('--max_length', type=int, default=41, help='path of the data lists')

        # inference
        parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
        parser.add_argument('--checkpoint', default='./models_ckpt/MCI_model.pth',
                            help='path to checkpoint')
        parser.add_argument('--result_path', default="./predict_result/",
                            help='path to save the result of masks and captions')

        # SAM 感知帧原料（式(14) 的区域项 + 背景项）
        parser.add_argument('--sam_frame_path', default=None,
                            help='预计算的 SAM 帧原料 .pt（形如 [2,3,H,W]，或 {"region":..., "bg":...}）；'
                                 '不指定则现场用 SAM 生成')
        parser.add_argument('--sam_folder', default='./saved_sam_frames',
                            help='SAM 缓存目录，用于查找在线生成所需的提取器权重 sam_frame_extractor.pth')
        parser.add_argument('--sam_weights_path', default=None,
                            help='在线生成 SAM 帧时使用的提取器权重；默认取 --sam_folder/sam_frame_extractor.pth')

        # backbone parameters
        parser.add_argument('--network', default='segformer-mit_b1',
                            help='define the backbone encoder to extract features')
        parser.add_argument('--encoder_dim', type=int, default=512,
                            help='the dimension of extracted features using backbone ')
        parser.add_argument('--feat_size', type=int, default=16,
                            help='define the output size of encoder to extract features')
        parser.add_argument('--dropout', type=float, default=0.1)
        # Model parameters
        parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
        parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in AttentionEncoder.')
        parser.add_argument('--decoder_n_layers', type=int, default=1)
        parser.add_argument('--feature_dim', type=int, default=512, help='embedding dimension')
        # X3D video encoder
        parser.add_argument('--in_height', type=int, default=256, help='Input image height for X3D')
        parser.add_argument('--in_width', type=int, default=256, help='Input image width for X3D')
        parser.add_argument('--pretrained', type=str, default=None,
                            help='X3D 预训练权重；推理时留空即可（权重来自 checkpoint）')

        args = parser.parse_args()

        return args

    def __init__(self, args=None):
        """
        Training and validation.
        """
        # 未传入 args 时退回命令行解析（保持 tools.py 等旧调用方式可用）
        if args is None:
            args = self.define_args()
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

        self.mean = [0.39073 * 255, 0.38623 * 255, 0.32989 * 255]
        self.std = [0.15329 * 255, 0.14628 * 255, 0.13648 * 255]

        # 使用传入的 args 中的路径
        vocab_path = os.path.join(args.list_path, args.vocab_file + '.json')
        with open(vocab_path, 'r') as f:
            self.word_vocab = json.load(f)

        # Load checkpoint
        snapshot_full_path = args.checkpoint
        print(f"Loading checkpoint from: {snapshot_full_path}")
        checkpoint = torch.load(snapshot_full_path, map_location=self.device)

        self.encoder = SAMEncoder(args).to(self.device)

        # 探测 X3D 实际输出的通道数，避免用固定的 --encoder_dim 拼错网络
        with torch.no_grad():
            dummy = torch.randn(1, 3, args.in_height, args.in_width, device=self.device)
            cd_feature, cc_feature = self.encoder(dummy, dummy)
            cd_channels, cc_channels = cd_feature.shape[1], cc_feature.shape[1]
        print(f"cd_feature channels: {cd_channels}, cc_feature channels: {cc_channels}")

        self.encoder_trans = AttentiveEncoder(
            input_cd_channels=cd_channels,
            input_cc_channels=cc_channels,
            seg_channels=64,
            num_classes=3,
            text_dim=args.feature_dim
        ).to(self.device)

        self.decoder = DecoderTransformer(encoder_dim=cc_channels, feature_dim=args.feature_dim,
                                          vocab_size=len(self.word_vocab), max_lengths=args.max_length,
                                          word_vocab=self.word_vocab, n_head=args.n_heads,
                                          n_layers=args.decoder_n_layers, dropout=args.dropout).to(self.device)

        self.encoder.load_state_dict(checkpoint['encoder_dict'])
        missing, unexpected = self.encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'], strict=False)
        if missing or unexpected:
            print(f"[警告] encoder_trans 权重与当前结构不完全一致：missing={list(missing)} "
                  f"unexpected={list(unexpected)}")
        self.decoder.load_state_dict(checkpoint['decoder_dict'])

        # Move to GPU, if available
        self.encoder.eval()
        self.encoder_trans.eval()
        self.decoder.eval()

    # ==================== SAM 感知帧原料（式(14) 的区域项与背景项） ====================
    def _load_sam_ingredients(self, sam_frame_path):
        blob = torch.load(sam_frame_path, map_location='cpu')
        if isinstance(blob, dict):
            f_reg, s_proj = blob['region'], blob['bg']
        elif torch.is_tensor(blob) and blob.dim() == 4 and blob.shape[0] == 2:
            f_reg, s_proj = blob[0], blob[1]
        else:
            raise ValueError(
                f'无法识别的 SAM 帧缓存格式：{type(blob)} / {tuple(getattr(blob, "shape", ()))}，'
                '期望 [2, C, H, W] 或 {"region": ..., "bg": ...}。'
                '请用 preprocess_sam_frames.py 重新生成缓存。'
            )
        return (f_reg.float().unsqueeze(0).to(self.device),
                s_proj.float().unsqueeze(0).to(self.device))

    def _generate_sam_ingredients(self, path_A, path_B):
        """没有预计算缓存时现场跑 SAM 建帧。

        需要 segment_anything / lightglue 以及 SAM 权重，因此这里的 import 是懒加载的：
        只用缓存推理时不会付出加载 SAM-H 的代价。
        """
        from model.SAMframe import (sam, mask_generator, matcher, RegionHead,
                                    SAMFrameExtractor, build_sam_frames)

        weights_path = self.args.sam_weights_path or os.path.join(self.args.sam_folder,
                                                                  'sam_frame_extractor.pth')
        region_head = RegionHead(in_channels=256, hidden=128, out_dim=256).to(self.device)
        frame_extractor = SAMFrameExtractor(d=256, C=3, H=self.args.in_height,
                                            W=self.args.in_width).to(self.device)

        if os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location=self.device)
            region_head.load_state_dict(ckpt['region_head'])
            frame_extractor.load_state_dict(ckpt['frame_extractor'])
            print(f"已加载 SAM 提取器权重：{weights_path}")
        else:
            print(f"[警告] 未找到 SAM 提取器权重 {weights_path}，将使用随机初始化。"
                  "训练缓存用的是另一次随机初始化，因此结果会与训练/论文不一致；"
                  "请先运行 preprocess_sam_frames.py 生成缓存与权重。")

        sam.to(self.device)
        matcher.to(self.device)

        with torch.no_grad():
            f_reg, s_proj = build_sam_frames(path_A, path_B, mask_generator, sam, region_head, matcher,
                                             frame_extractor, self.device,
                                             d=256, C=3, H=self.args.in_height, W=self.args.in_width)

        return f_reg.unsqueeze(0).to(self.device), s_proj.unsqueeze(0).to(self.device)

    def get_sam_ingredients(self, path_A, path_B):
        """优先读预计算缓存，否则现场生成。"""
        if self.args.sam_frame_path is not None:
            return self._load_sam_ingredients(self.args.sam_frame_path)
        return self._generate_sam_ingredients(path_A, path_B)

    def preprocess(self, path_A, path_B):

        imgA = imread(path_A)
        imgB = imread(path_B)
        imgA = np.asarray(imgA, np.float32)
        imgB = np.asarray(imgB, np.float32)

        imgA = imgA.transpose(2, 0, 1)
        imgB = imgB.transpose(2, 0, 1)
        for i in range(len(self.mean)):
            imgA[i, :, :] -= self.mean[i]
            imgA[i, :, :] /= self.std[i]
            imgB[i, :, :] -= self.mean[i]
            imgB[i, :, :] /= self.std[i]

        if imgA.shape[1] != 256 or imgA.shape[2] != 256:
            imgA = cv2.resize(imgA, (256, 256))
            imgB = cv2.resize(imgB, (256, 256))

        imgA = torch.FloatTensor(imgA)
        imgB = torch.FloatTensor(imgB)
        imgA = imgA.unsqueeze(0)  # (1, 3, 256, 256)
        imgB = imgB.unsqueeze(0)

        return imgA.to(self.device), imgB.to(self.device)

    @torch.no_grad()
    def infer(self, path_A, path_B):
        """完整推理流程：SP-TOPE 编码 → S²MG 双向引导 → 变化掩码 + 变化描述。

        与 validation 中的流程保持一致：
        CD 先用粗变化概率引导字幕生成，再从句子里抽变化实体反过来精化检测。
        """
        print('model_infer: start')
        imgA, imgB = self.preprocess(path_A, path_B)
        sam_region, sam_bg = self.get_sam_ingredients(path_A, path_B)

        # SP-TOPE：四帧序列 [I1, I_random, I_SAM, I2] 的视频编码
        cd_feature, cc_feature = self.encoder(imgA, imgB, sam_region=sam_region, sam_bg=sam_bg)
        seg_initial, cc_feature = self.encoder_trans(cd_feature, cc_feature)

        # CD → CC：用粗变化概率作为字幕生成的空间先验
        seg_prob = torch.softmax(seg_initial, dim=1)
        change_prob = torch.clamp(seg_prob[:, 1:2] + seg_prob[:, 2:3], 0, 1)
        change_mask = F.interpolate(change_prob, size=cc_feature.shape[-2:],
                                    mode='bilinear', align_corners=False)
        generated = self.decoder.sample_tokens(cc_feature, change_mask=change_mask)

        # CC → CD：用生成句中的变化实体反向精化变化区域
        entity_emb, entity_valid = self.decoder.encode_entities(generated)
        seg = self.encoder_trans.refine(cd_feature, seg_initial, entity_emb, entity_valid)

        # 结果：argmax 得到变化掩码，反查词表得到描述
        pred_mask = np.argmax(seg.data.cpu().numpy(), axis=1)[0].astype(np.uint8)
        pred_seq = [w for w in generated.squeeze(0).tolist()
                    if w not in {self.word_vocab['<START>'], self.word_vocab['<END>'],
                                 self.word_vocab['<NULL>']}]
        caption = ' '.join(list(self.word_vocab.keys())[i] for i in pred_seq)

        return pred_mask, caption

    def generate_change_caption(self, path_A, path_B):
        _, caption = self.infer(path_A, path_B)
        print('change captioning:', caption)
        return caption

    def change_detection(self, path_A, path_B, savepath_mask):
        pred, _ = self.infer(path_A, path_B)

        # 保存图片：道路=黄，建筑=红，背景=黑
        pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        pred_rgb[pred == 1] = [0, 255, 255]
        pred_rgb[pred == 2] = [0, 0, 255]

        save_dir = os.path.dirname(os.path.abspath(savepath_mask))
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(savepath_mask, pred_rgb)
        print('model_infer: mask saved in', savepath_mask)

        return pred  # (256,256) 取值 {0,1,2}

    def compute_object_num(self, changed_mask, object):
        print("compute num start")
        # compute the number of connected components
        mask = changed_mask
        mask_cp = 0 * mask.copy()
        if object == 'road':
            mask_cp[mask == 1] = 255
        elif object == 'building':
            mask_cp[mask == 2] = 255
        lbl = measure.label(mask_cp, connectivity=2)
        props = measure.regionprops(lbl)
        # get bboxes by a for loop
        bboxes = []
        for prop in props:
            if prop.area > 5:
                bboxes.append([prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]])
        num = len(bboxes)
        print('Found', num, object)
        print('compute num end')
        num_str = 'Found ' + str(num) + ' changed ' + object
        return num_str

    # design more tool functions:


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Interpretation')
    parser.add_argument('--imgA_path', type=str, required=True, help='Path to image A')
    parser.add_argument('--imgB_path', type=str, required=True, help='Path to image B')
    parser.add_argument('--mask_save_path', type=str, default='./CDmask.png', help='Path to save mask')
    parser.add_argument('--data_folder', default='./Levir-MCI-dataset/images', help='folder with data files')
    parser.add_argument('--list_path', default='./data/LEVIR_MCI', help='path of the data lists')
    parser.add_argument('--vocab_file', default='vocab', help='vocab file name')
    parser.add_argument('--max_length', type=int, default=41, help='max length of caption')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--checkpoint', default='./models_ckpt/MCI_model.pth', help='path to checkpoint')
    parser.add_argument('--result_path', default="./predict_result/", help='path to save results')
    parser.add_argument('--sam_frame_path', default=None,
                        help='预计算的 SAM 帧原料 .pt；不指定则现场用 SAM 生成')
    parser.add_argument('--sam_folder', default='./saved_sam_frames',
                        help='SAM 缓存目录，用于查找在线生成所需的提取器权重')
    parser.add_argument('--sam_weights_path', default=None,
                        help='在线生成 SAM 帧时使用的提取器权重')
    parser.add_argument('--network', default='segformer-mit_b1', help='backbone network')
    parser.add_argument('--encoder_dim', type=int, default=512, help='encoder dimension')
    parser.add_argument('--feat_size', type=int, default=16, help='feature size')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--n_heads', type=int, default=8, help='number of heads')
    parser.add_argument('--n_layers', type=int, default=3, help='number of layers in encoder')
    parser.add_argument('--decoder_n_layers', type=int, default=1, help='number of layers in decoder')
    parser.add_argument('--feature_dim', type=int, default=512, help='feature dimension')
    parser.add_argument('--in_height', type=int, default=256, help='Input image height for X3D')
    parser.add_argument('--in_width', type=int, default=256, help='Input image width for X3D')
    parser.add_argument('--pretrained', type=str, default=None, help='X3D 预训练权重，推理时留空')

    # 2. 解析参数
    args = parser.parse_args()

    # 3. 实例化类，传入解析好的 args
    change_agent = Change_Perception(args=args)

    # 4. 执行推理
    change_agent.generate_change_caption(args.imgA_path, args.imgB_path)
    change_agent.change_detection(args.imgA_path, args.imgB_path, args.mask_save_path)
