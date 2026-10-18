import torch
import torch.optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils import data
import torch.backends.cudnn as cudnn
import random
import numpy as np
import argparse
import json
import os
import time
from tqdm import tqdm
from data.LEVIR_MCI import SAMLEVIRCCDataset
from model.model_encoder_att import SAMEncoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer
from utils_tool.utils import *
from utils_tool.metrics import Evaluator
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer(object):
    def __init__(self, args):
        """
        Training and validation.
        """
        self.args = args
        self.resumed = False
        with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
            self.word_vocab = json.load(f)
        
        # ==================== Resume 核心逻辑 1/5: 初始化或恢复 ====================
        if args.resume or args.resume_path is not None:
            self._load_checkpoint()
            self.resumed = True
        else:
            # 全新训练初始化
            self.start_train_goal = args.train_goal
            self.best_bleu4 = 0.4  # BLEU-4 score right now
            self.MIou = 0.4
            self.Sum_Metric = 0.4
            self.start_epoch = 0
            self.best_model_path = None
            self.best_epoch = 0
            self.current_goal_index = 0  # 用于追踪多阶段训练 [2,1,0] 的进度
            random_str = str(random.randint(10, 100))
            name = 'baseline_'+time_file_str() + f'_train_goal_{args.train_goal}_' + random_str
            self.args.savepath = os.path.join(args.savepath, name)
            if os.path.exists(self.args.savepath)==False:
                os.makedirs(self.args.savepath)

        # 初始化 Log (Resume模式下追加写入)
        log_mode = 'a' if self.resumed else 'w'
        self.log = open(os.path.join(self.args.savepath, 'training.log'), log_mode)
        
        print_log('=>datset: {}'.format(args.data_name), self.log)
        print_log('=>network: {}'.format(args.network), self.log)
        print_log('=>encoder_lr: {}'.format(args.encoder_lr), self.log)
        print_log('=>decoder_lr: {}'.format(args.decoder_lr), self.log)
        print_log('=>num_epochs: {}'.format(args.num_epochs), self.log)
        print_log('=>train_batchsize: {}'.format(args.train_batchsize), self.log)

        
      

        # 构建模型 (如果是Resume，_load_checkpoint内部已经处理了权重加载)
        if not self.resumed:
            self.build_model()

        # Loss function
        self.criterion_cap = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
        # 论文式(32)：检测损失为普通交叉熵，不加类别权重
        self.criterion_det = torch.nn.CrossEntropyLoss().to(device)

        # Custom dataloaders

        #插入SAM帧的数据集加载
        if args.data_name == 'LEVIR_MCI':
            sam_folder = args.sam_folder
            self.train_loader = data.DataLoader(
                SAMLEVIRCCDataset(args.data_folder, args.list_path, 'train', args.token_folder, args.vocab_file, args.max_length, args.allow_unk, sam_frame_folder=sam_folder),
                batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
            self.val_loader = data.DataLoader(
                SAMLEVIRCCDataset(args.data_folder, args.list_path, 'val', args.token_folder, args.vocab_file, args.max_length, args.allow_unk, sam_frame_folder=sam_folder),
                batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)

        self.index_i = 0
        self.hist = np.zeros((args.num_epochs*2 * len(self.train_loader), 7))
        # Epochs

        self.evaluator = Evaluator(num_class=3)

    # ==================== Resume 核心逻辑 2/5: 保存 Checkpoint ====================
    def save_checkpoint(self, epoch, is_best=False):
        """
        保存完整的训练状态，包括模型、优化器、调度器和指标
        """
        state = {
            # 1. 模型权重
            'encoder_dict': self.encoder.state_dict(),
            'encoder_trans_dict': self.encoder_trans.state_dict(),
            'decoder_dict': self.decoder.state_dict(),
            
            # 2. 优化器状态
            'encoder_optimizer': self.encoder_optimizer.state_dict() if self.encoder_optimizer is not None else None,
            'encoder_trans_optimizer': self.encoder_trans_optimizer.state_dict(),
            'decoder_optimizer': self.decoder_optimizer.state_dict() if self.decoder_optimizer is not None else None,
            
            # 3. 学习率调度器状态
            'encoder_lr_scheduler': self.encoder_lr_scheduler.state_dict() if self.encoder_lr_scheduler is not None else None,
            'encoder_trans_lr_scheduler': self.encoder_trans_lr_scheduler.state_dict(),
            'decoder_lr_scheduler': self.decoder_lr_scheduler.state_dict() if self.decoder_lr_scheduler is not None else None,
            
            # 4. 训练进度与指标
            'epoch': epoch,
            'best_bleu4': self.best_bleu4,
            'MIou': self.MIou,
            'Sum_Metric': self.Sum_Metric,
            'best_epoch': self.best_epoch,
            'best_model_path': self.best_model_path,
            
            # 5. 训练阶段配置 (用于多阶段恢复)
            'start_train_goal': self.start_train_goal,
            'current_train_goal': self.args.train_goal,
            'train_stage': self.args.train_stage,
            'current_goal_index': getattr(self, 'current_goal_index', 0),
            'savepath': self.args.savepath,
        }

        # 始终保存最新的 Checkpoint (覆盖式)
        latest_path = os.path.join(self.args.savepath, 'checkpoint_latest.pth')
        torch.save(state, latest_path)
        print_log(f"Checkpoint saved to {latest_path}", self.log)

        # 如果是 Best Model，另外保存一份
        if is_best:
            if epoch > 10:
                metric = f'Sum_{round(100000 * self.Sum_Metric)}_MIou_{round(100000 * self.MIou)}_Bleu4_{round(100000 * self.best_bleu4)}'
                model_name = f'{self.args.data_name}_bts_{self.args.train_batchsize}_{self.args.network}_epo_{epoch}_{metric}.pth'
                best_path = os.path.join(self.args.savepath, model_name)
                torch.save(state, best_path)
                self.best_model_path = best_path
                print_log(f"New Best Model saved to {best_path}", self.log)
            else:
                # 前 10 个 epoch 不另存带 epoch 命名的文件，但 best_model_path 必须指向
                # 一个真实存在的文件：s2 阶段会 torch.load(best_model_path)，指向未落盘的
                # 文件会直接 FileNotFoundError
                self.best_model_path = latest_path

    # ==================== Resume 核心逻辑 3/5: 加载 Checkpoint ====================
    def _load_checkpoint(self):
        # 1. 确定 Checkpoint 路径
        if self.args.resume_path is not None:
            checkpoint_path = self.args.resume_path
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"指定的 Resume 路径不存在: {checkpoint_path}")
        else:
            # 自动在 savepath 下寻找最新的 .pth 文件
            if not os.path.exists(self.args.savepath):
                raise FileNotFoundError(f"Resume 目录不存在: {self.args.savepath}")
            
            # 查找所有 pth 文件并按修改时间排序
            files = [f for f in os.listdir(self.args.savepath) if f.endswith('.pth')]
            if not files:
                raise FileNotFoundError(f"在 {self.args.savepath} 中未找到任何 .pth 文件")
            
            files.sort(key=lambda x: os.path.getmtime(os.path.join(self.args.savepath, x)), reverse=True)
            checkpoint_path = os.path.join(self.args.savepath, files[0])

        print(f"[Resume] 正在加载: {checkpoint_path}")
        
        # 2. 加载字典
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 3. 恢复基础状态
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_bleu4 = checkpoint['best_bleu4']
        self.MIou = checkpoint['MIou']
        self.Sum_Metric = checkpoint['Sum_Metric']
        self.best_epoch = checkpoint['best_epoch']
        self.best_model_path = checkpoint.get('best_model_path', None)
        
        # 恢复训练阶段
        self.start_train_goal = checkpoint['start_train_goal']
        self.args.train_goal = checkpoint['current_train_goal']
        self.args.train_stage = checkpoint['train_stage']
        self.current_goal_index = checkpoint.get('current_goal_index', 0)
        
        # 关键：恢复 savepath，防止创建新目录
        self.args.savepath = checkpoint['savepath']
        print(f"[Resume] 已恢复到 Epoch {self.start_epoch}, 当前目标 train_goal={self.args.train_goal}")

        # 4. 构建模型结构
        # Build architecture/optimizers without loading an s1 checkpoint again.
        self.build_model(restoring=True)
        
        # 5. 恢复模型权重
        self.encoder.load_state_dict(checkpoint['encoder_dict'])
        self.encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_dict'])

        # 6. 恢复优化器状态
        if self.encoder_optimizer is not None and checkpoint.get('encoder_optimizer') is not None:
            self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
        if checkpoint.get('encoder_trans_optimizer') is not None:
            self.encoder_trans_optimizer.load_state_dict(checkpoint['encoder_trans_optimizer'])
        if self.decoder_optimizer is not None and checkpoint.get('decoder_optimizer') is not None:
            self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])

        # 7. 恢复学习率调度器状态
        if self.encoder_lr_scheduler is not None and checkpoint.get('encoder_lr_scheduler') is not None:
            self.encoder_lr_scheduler.load_state_dict(checkpoint['encoder_lr_scheduler'])
        if checkpoint.get('encoder_trans_lr_scheduler') is not None:
            self.encoder_trans_lr_scheduler.load_state_dict(checkpoint['encoder_trans_lr_scheduler'])
        if self.decoder_lr_scheduler is not None and checkpoint.get('decoder_lr_scheduler') is not None:
            self.decoder_lr_scheduler.load_state_dict(checkpoint['decoder_lr_scheduler'])

        print("[Resume] 模型、优化器、学习率调度器恢复成功。")

    def build_model(self, restoring=False):
        args = self.args
        if args.train_stage not in {'s1', 's2'}:
            raise ValueError('train_stage must be s1 or s2')
        if args.train_stage == 's2' and args.checkpoint is None and not restoring:
            raise ValueError('s2 requires --checkpoint')
        if args.train_stage == 's1' or not hasattr(self, 'encoder'):
            self.encoder = SAMEncoder(args)
            self.encoder = self.encoder.to(device) 
            self.encoder.fine_tune(args.fine_tune_encoder)

            with torch.no_grad():
                h_in = args.in_height if hasattr(args, 'in_height') else 256
                w_in = args.in_width if hasattr(args, 'in_width') else 256
                dummy_imgA = torch.randn(1, 3, h_in, w_in).to(device)
                dummy_imgB = torch.randn(1, 3, h_in, w_in).to(device)
                cd_feature, cc_feature = self.encoder(dummy_imgA, dummy_imgB)
            
                cd_channels = cd_feature.shape[1]
                cc_channels = cc_feature.shape[1]
                # 假设 cc_feature 的空间尺寸是 H_cc x W_cc
                _, _, h_cc, w_cc = cc_feature.shape


            self.encoder_trans = AttentiveEncoder(
                    input_cd_channels=cd_channels,
                    input_cc_channels=cc_channels,
                    seg_channels=64,      # 中间层通道
                    num_classes=3,        # 分割类别数
                    text_dim=args.feature_dim
            )
            self.decoder = DecoderTransformer(encoder_dim=cc_channels, feature_dim=args.feature_dim,
                                              vocab_size=len(self.word_vocab), max_lengths=args.max_length,
                                              word_vocab=self.word_vocab, n_head=args.n_heads,
                                              n_layers=args.decoder_n_layers, dropout=args.dropout)

            fine_tune_capdecoder = True
        if args.train_stage == 's2':
            if not restoring:
                # 非Resume：正常加载s1的最优权重
                checkpoint = torch.load(args.checkpoint, map_location=device)
                print('Load Model from {}'.format(args.checkpoint))
                self.decoder.load_state_dict(checkpoint['decoder_dict'])
                self.encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'])
                self.encoder.load_state_dict(checkpoint['encoder_dict'])
            # eval()
            self.encoder.eval()
            self.encoder_trans.eval()
            self.decoder.eval()
            # 各个modules 是否需要微调
            args.fine_tune_encoder = False
            self.encoder.fine_tune(args.fine_tune_encoder)
            self.encoder_trans.fine_tune(args.train_goal)
            fine_tune_capdecoder = False if args.train_goal == 0 else True
            self.decoder.fine_tune(fine_tune_capdecoder)

        # set optimizer
        self.encoder_optimizer = torch.optim.Adam(params=self.encoder.parameters(),
                                                  lr=args.encoder_lr) if args.fine_tune_encoder else None
        self.encoder_trans_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.encoder_trans.parameters()),
            lr=args.encoder_lr)
        self.decoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.decoder.parameters()),
            lr=args.decoder_lr) if fine_tune_capdecoder else None

        # Move to GPU, if available
        
        self.encoder_trans = self.encoder_trans.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_optimizer, step_size=5,
                                                                    gamma=1.0) if args.fine_tune_encoder else None
        self.encoder_trans_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_trans_optimizer, step_size=5,
                                                                          gamma=1.0)
        self.decoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.decoder_optimizer, step_size=5,
                                                                    gamma=1.0) if fine_tune_capdecoder else None

    
    #学习率调整函数
    def adjust_learning_rate(self, optimizer, shrink_factor=0.5):
        if optimizer is None:
            return
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * shrink_factor
        print_log(f"学习率已衰减 → 新学习率: {optimizer.param_groups[0]['lr']:.8f}", self.log)

    # ===================== 打印当前学习率函数 =====================
    def print_current_lr(self, epoch):
        print_log(f"\n========== Epoch {epoch} learning rate ==========", self.log)
        if self.encoder_optimizer is not None:
            enc_lr = self.encoder_optimizer.param_groups[0]['lr']
            print_log(f"Encoder learning rate: {enc_lr:.8f}", self.log)
        trans_lr = self.encoder_trans_optimizer.param_groups[0]['lr']
        print_log(f"Encoder_Trans learning rate: {trans_lr:.8f}", self.log)
        if self.decoder_optimizer is not None:
            dec_lr = self.decoder_optimizer.param_groups[0]['lr']
            print_log(f"Decoder learning rate: {dec_lr:.8f}", self.log)
        print_log("=========================================\n", self.log)
    
    
    def training(self, args, epoch):
        self.print_current_lr(epoch)
        # ===================== 每10轮衰减学习率 =====================
        if epoch > 0 and epoch % 10 == 0:
            self.adjust_learning_rate(self.encoder_optimizer, shrink_factor=0.5)
            self.adjust_learning_rate(self.encoder_trans_optimizer, shrink_factor=0.5)
            self.adjust_learning_rate(self.decoder_optimizer, shrink_factor=0.5)
        
        if self.args.train_stage == 's1':
            self.encoder.train()
            self.encoder_trans.train()
            self.decoder.train()  # train mode (dropout and batchnorm is used)
        else:
            # Honor staged freezing for both [2,1,0] and standalone s2 runs.
            # requires_grad alone does not freeze BatchNorm running statistics.
            self.encoder.eval()
            self.encoder_trans.eval()
            self.encoder_trans.fine_tune(self.args.train_goal)
            self.decoder.train(self.args.train_goal == 1)

        if self.decoder_optimizer is not None:
            self.decoder_optimizer.zero_grad()
        self.encoder_trans_optimizer.zero_grad()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()
        for id, (imgA, imgB, seg_label, _, _, token, token_len, _, sam_region, sam_bg) in enumerate(self.train_loader):
            # if id == 120:
            #    break
            start_time = time.time()
            accum_steps = 64//args.train_batchsize

            # Move to GPU, if available
            imgA = imgA.to(device)
            imgB = imgB.to(device)
            seg_label = seg_label.to(device)
            token = token.squeeze(1).to(device)
            token_len = token_len.to(device)
            sam_region = sam_region.to(device)
            sam_bg = sam_bg.to(device)
            # Forward prop.
            cd_feature, cc_feature = self.encoder(imgA, imgB, sam_region=sam_region, sam_bg=sam_bg)
            seg_initial, cc_feature = self.encoder_trans(cd_feature, cc_feature) # 得到变化检测结果
            seg = seg_initial
            b, c, h, w = cc_feature.shape
            change_mask = None

            cap_loss = torch.tensor(0.0).to(device)
            raw_cap_loss = 0.0
            # CD-only fine-tuning after joint training still uses frozen CC
            # predictions for semantic feedback, just as validation does.
            use_feedback = self.args.train_goal != 0 or self.start_train_goal == 2 or self.args.train_stage == 's2'
            if use_feedback:
                seg_prob = torch.softmax(seg_initial, dim=1)
                change_prob = seg_prob[:, 1:2] + seg_prob[:, 2:3]
                # change_prob = seg_prob[:, 1:2] 
                change_prob = torch.clamp(change_prob, 0, 1)
                target_h, target_w = cc_feature.shape[-2], cc_feature.shape[-1]
                change_mask = F.interpolate(
                                change_prob, 
                                size=(target_h, target_w), 
                                mode='bilinear', 
                                align_corners=False
                            )
                if self.args.train_stage == 's2' and self.args.train_goal == 1:
                    change_mask = change_mask.detach()
                    
            if self.args.train_goal != 0:
                scores, caps_sorted, decode_lengths, sort_ind = self.decoder(cc_feature, token, token_len, change_mask=change_mask)
                # scores, caps_sorted, decode_lengths, sort_ind = self.decoder(cc_feature, token, token_len)
                # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                targets = caps_sorted[:, 1:]
                scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
                # Calculate loss
                cap_loss = self.criterion_cap(scores, targets.to(torch.int64))
                raw_cap_loss = cap_loss.detach().item()
            if use_feedback:
                generated = self.decoder.sample_tokens(cc_feature, change_mask=change_mask)
                entity_emb, entity_valid = self.decoder.encode_entities(generated)
                seg = self.encoder_trans.refine(cd_feature, seg_initial, entity_emb, entity_valid)
            # 论文式(32)：只在最终变化图（经 S2MG 精化后的 seg）上计算交叉熵
            det_loss = self.criterion_det(seg, seg_label.to(torch.int64))
            raw_det_loss = det_loss.detach().item()
            
            if self.args.train_goal == 0:
                if self.start_train_goal==2:
                    if epoch < 100:
                        det_loss = det_loss #/ det_loss.detach().item()
                loss = det_loss
            elif self.args.train_goal == 1:
                # if args.train_stage == 's1':
                #     cap_loss = cap_loss / cap_loss.detach().item()
                loss = cap_loss
            else:
                # balance two losses
                if args.train_stage == 's1':
                    if not torch.isfinite(det_loss):
                        print_log(f"det loss is {det_loss.item()}, skipping update", self.log)
                        det_loss = torch.tensor(0.0).to(device)
                    if not torch.isfinite(cap_loss):
                        print_log(f"cap loss is {cap_loss.item()}, skipping update", self.log)
                        cap_loss = torch.tensor(0.0).to(device)

                    if raw_det_loss < 1e-8 or raw_cap_loss < 1e-8:
                        print_log(f"det loss is {raw_det_loss}, cap loss is {raw_cap_loss}", self.log)
                    # 论文式(34)：分母 detach(L) 当作常数，梯度只保留在分子上
                    det_loss = det_loss / det_loss.detach().clamp_min(1e-8)
                    cap_loss = cap_loss / cap_loss.detach().clamp_min(1e-8)

                loss = det_loss + cap_loss
            # Back prop.
            loss = loss / accum_steps
            loss.backward()
            # Clip gradients
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=args.grad_clip)
                torch.nn.utils.clip_grad_norm_(self.encoder_trans.parameters(), max_norm=args.grad_clip)
                if self.encoder_optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=args.grad_clip) 

            # Update weights
            if (id + 1) % accum_steps == 0 or (id + 1) == len(self.train_loader):
                if self.decoder_optimizer is not None:
                    self.decoder_optimizer.step()
                self.encoder_trans_optimizer.step()
                if self.encoder_optimizer is not None:
                    # if epoch >10:
                    self.encoder_optimizer.step()

                # Adjust learning rate
                if self.decoder_lr_scheduler is not None:
                    self.decoder_lr_scheduler.step()
                # print(decoder_optimizer.param_groups[0]['lr'])
                self.encoder_trans_lr_scheduler.step()
                if self.encoder_lr_scheduler is not None:
                    # if epoch > 10:
                    self.encoder_lr_scheduler.step()
                    # print(encoder_optimizer.param_groups[0]['lr'])

                if self.decoder_optimizer is not None:
                    self.decoder_optimizer.zero_grad()
                self.encoder_trans_optimizer.zero_grad()
                if self.encoder_optimizer is not None:
                    self.encoder_optimizer.zero_grad()

            # Keep track of metrics
            self.hist[self.index_i, 0] = time.time() - start_time #batch_time
            if self.args.train_goal == 0 or self.args.train_goal == 2:
                self.hist[self.index_i, 1] = det_loss.item() #train_loss
                self.hist[self.index_i, 2] = accuracy(seg.permute(0, 2, 3, 1).reshape(-1, seg.size(1)),
                                                      seg_label.reshape(-1), 1)
                self.hist[self.index_i, 5] = raw_det_loss
            if self.args.train_goal == 1 or self.args.train_goal == 2:
                self.hist[self.index_i, 3] = cap_loss.item()  # train_loss
                if scores is not None and targets is not None:
                    self.hist[self.index_i, 4] = accuracy(scores, targets, 5) #top5
                self.hist[self.index_i, 6] = raw_cap_loss

            self.index_i += 1
            # Print status
            if self.index_i % args.print_freq == 0:
                print_log('Training Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time: {3:.3f}\t'
                    'Det_Loss: {4:.4f}\t'
                    'Det Acc: {5:.3f}\t'
                    'Cap_loss: {6:.5f}\t'
                    'Text_Top-5 Acc: {7:.3f}\t'
                    'Raw_Det_Loss: {8:.4f}\t'
                    'Raw_Cap_Loss: {9:.5f}'
                    .format(epoch, id, len(self.train_loader),
                                        np.mean(self.hist[self.index_i-args.print_freq:self.index_i-1,0])*args.print_freq,
                                        np.mean(self.hist[self.index_i-args.print_freq:self.index_i-1,1]),
                                        np.mean(self.hist[self.index_i-args.print_freq:self.index_i-1,2]),
                                        np.mean(self.hist[self.index_i-args.print_freq:self.index_i-1,3]),
                                        np.mean(self.hist[self.index_i-args.print_freq:self.index_i-1,4]),
                                        np.mean(self.hist[self.index_i-args.print_freq:self.index_i-1,5]),
                                        np.mean(self.hist[self.index_i-args.print_freq:self.index_i-1,6])
                                ), self.log)

    # One epoch's validation
    def validation(self, epoch):
        word_vocab = self.word_vocab
        self.decoder.eval()  # eval mode (no dropout or batchnorm)
        self.encoder_trans.eval()
        if self.encoder is not None:
            self.encoder.eval()

        val_start_time = time.time()
        references = list()  # references (true captions) for calculating BLEU-4 score
        hypotheses = list()  # hypotheses (predictions)

        self.evaluator.reset()
        with torch.no_grad():
            # Batches
            for ind, (imgA, imgB, seg_label, token_all, token_all_len, _, _, _, sam_region, sam_bg) in enumerate(
                    tqdm(self.val_loader, desc='val_' + "EVALUATING AT BEAM SIZE " + str(1))):
                # Move to GPU, if available
                imgA = imgA.to(device)
                imgB = imgB.to(device)
                sam_region = sam_region.to(device)
                sam_bg = sam_bg.to(device)
                token_all = token_all.squeeze(0).to(device)
                # Forward prop.
                if self.encoder is not None:
                    cd_feature, cc_feature = self.encoder(imgA, imgB, sam_region=sam_region, sam_bg=sam_bg)
                seg_initial, cc_feature = self.encoder_trans(cd_feature, cc_feature)
                seg = seg_initial
                
                change_mask_val = None
                use_feedback = self.args.train_goal != 0 or self.start_train_goal == 2 or self.args.train_stage == 's2'
                if use_feedback:
                    seg_prob_val = torch.softmax(seg_initial, dim=1)
                    change_prob_val = seg_prob_val[:, 1:2] + seg_prob_val[:, 2:3]
                    # change_prob_val = seg_prob_val[:, 1:2] 
                    change_prob_val = torch.clamp(change_prob_val, 0, 1)
                    target_h, target_w = cc_feature.shape[-2], cc_feature.shape[-1]
                    change_mask_val = F.interpolate(
                        change_prob_val, 
                        size=(target_h, target_w), 
                        mode='bilinear', 
                        align_corners=False
                    )

                if use_feedback:
                    generated = self.decoder.sample_tokens(cc_feature, change_mask=change_mask_val)
                    entity_emb, entity_valid = self.decoder.encode_entities(generated)
                    seg = self.encoder_trans.refine(cd_feature, seg_initial, entity_emb, entity_valid)
                    seq = generated.squeeze(0).tolist()
                # for segmentation
                if self.args.train_goal != 1  or self.start_train_goal == 2:
                    pred_seg = seg.data.cpu().numpy()
                    seg_label = seg_label.cpu().numpy()
                    pred_seg = np.argmax(pred_seg, axis=1)
                    # Add batch sample into evaluator
                    self.evaluator.add_batch(seg_label, pred_seg)
                # for captioning
                if self.args.train_goal != 0 or self.start_train_goal == 2:
                    img_token = token_all.tolist()
                    img_tokens = list(map(lambda c: [w for w in c if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}],
                            img_token))  # remove <start> and pads
                    references.append(img_tokens)

                    pred_seq = [w for w in seq if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
                    hypotheses.append(pred_seq)
                    assert len(references) == len(hypotheses)

                    if ind % self.args.print_freq == 0:
                        pred_caption = ""
                        ref_caption = ""
                        for i in pred_seq:
                            pred_caption += (list(word_vocab.keys())[i]) + " "
                        ref_caption = ""
                        for i in img_tokens:
                            for j in i:
                                ref_caption += (list(word_vocab.keys())[j]) + " "
                            ref_caption += ".    "
            val_time = time.time() - val_start_time
            # Fast test during the training
            # for segmentation
            Acc_seg, Acc_class_seg, mIoU_seg, FWIoU_seg, IoU = 0.0, 0.0, 0.0, 0.0, []
            if self.args.train_goal != 1 or self.start_train_goal == 2:
                Acc_seg = self.evaluator.Pixel_Accuracy()
                Acc_class_seg = self.evaluator.Pixel_Accuracy_Class()
                mIoU_seg, IoU = self.evaluator.Mean_Intersection_over_Union()
                FWIoU_seg = self.evaluator.Frequency_Weighted_Intersection_over_Union()
                print_log(
                    '\nDetection_Validation:\n' 'Acc_seg: {0:.5f}\t' 'Acc_class_seg: {1:.5f}\t' 'mIoU_seg: {2:.5f}\t' 'FWIoU_seg: {3:.5f}\t '
                    .format(Acc_seg, Acc_class_seg, mIoU_seg, FWIoU_seg), self.log)
                print_log('Iou: {}'.format(IoU), self.log)

            # Calculate evaluation scores
            Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            if self.args.train_goal != 0 or self.start_train_goal == 2:
                score_dict = get_eval_score(references, hypotheses)
                Bleu_1 = score_dict['Bleu_1']
                Bleu_2 = score_dict['Bleu_2']
                Bleu_3 = score_dict['Bleu_3']
                Bleu_4 = score_dict['Bleu_4']
                Meteor = score_dict['METEOR']
                Rouge = score_dict['ROUGE_L']
                Cider = score_dict['CIDEr']
                print_log('Captioning_Validation:\n' 'Time: {0:.3f}\t' 'BLEU-1: {1:.5f}\t' 'BLEU-2: {2:.5f}\t' 'BLEU-3: {3:.5f}\t' 
                    'BLEU-4: {4:.5f}\t' 'Meteor: {5:.5f}\t' 'Rouge: {6:.5f}\t' 'Cider: {7:.5f}\t'
                    .format(val_time, Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider), self.log)

        # ==================== Resume 核心逻辑 4/5: 判断 Best 并保存 ====================
        is_best = False
        if self.start_train_goal != 2:
            if self.args.train_goal == 0:
                Bleu_4 = 0
            if self.args.train_goal == 1:
                mIoU_seg = 0
            if Bleu_4 > self.best_bleu4 or mIoU_seg > self.MIou or Bleu_4 + mIoU_seg > self.Sum_Metric:
                self.best_bleu4 = max(Bleu_4, self.best_bleu4)
                self.MIou = max(mIoU_seg, self.MIou)
                self.Sum_Metric = max(Bleu_4 + mIoU_seg, self.Sum_Metric)
                is_best = True
                self.best_epoch = epoch
                
        elif self.start_train_goal == 2:
            Sum_Metric = mIoU_seg + Bleu_4
            if (self.args.train_goal == 2 and Sum_Metric >= self.Sum_Metric) or \
               (self.args.train_goal == 1 and Bleu_4 >= self.best_bleu4) or \
               (self.args.train_goal == 0 and mIoU_seg > self.MIou):
                
                self.best_bleu4 = max(Bleu_4, self.best_bleu4) if self.args.train_goal == 1 else Bleu_4
                self.MIou = max(mIoU_seg, self.MIou) if self.args.train_goal == 0 else mIoU_seg
                self.Sum_Metric = max(Sum_Metric, self.Sum_Metric) if self.args.train_goal == 2 else Sum_Metric
                is_best = True
                self.best_epoch = epoch
                
                if is_best:
                    metric = f'Sum_{round(100000*self.Sum_Metric)}_MIou_{round(100000*self.MIou)}_Bleu4_{round(100000*self.best_bleu4)}'
                    model_name = f'{self.args.data_name}_bts_{self.args.train_batchsize}_{self.args.network}_epo_{epoch}_{metric}.pth'
                    self.best_model_path = os.path.join(self.args.savepath, model_name)

        # 关键调用：保存 Checkpoint
        self.save_checkpoint(epoch, is_best=is_best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Interpretation')

    # Data parameters
    parser.add_argument('--sys', default='linux', help='system win or linux')
    parser.add_argument('--data_folder', default='/media/xidian/xd132/CXH/change-agent/Multi_change/Levir-MCI-dataset/images',help='folder with data files')
    parser.add_argument('--sam_folder', default='/media/xidian/xd132/CXH/change-agent/Multi_change/saved_sam_frames',
                        help='folder with cached SAM frame ingredients，需与 preprocess_sam_frames.py 的 --save_dir 一致')
    parser.add_argument('--list_path', default='./data/LEVIR_MCI/', help='path of the data lists')
    parser.add_argument('--token_folder', default='./data/LEVIR_MCI/tokens/', help='folder with token files')
    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--max_length', type=int, default=41, help='path of the data lists')
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    parser.add_argument('--data_name', default="LEVIR_MCI",help='base name shared by data files.')

    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint from stage s1, assert not None when train_stage=s2')
    parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches')
    
    # Training parameters
    parser.add_argument('--train_goal', type=int, default=2, help='0:det; 1:cap; 2:two tasks')
    parser.add_argument('--train_stage', default='s1', help='s1: pretrain backbone under two loss;'
                                                                         ' s2: train two branch respectively')
    parser.add_argument('--fine_tune_encoder', type=bool, default=True, help='whether fine-tune encoder or not')
    parser.add_argument('--train_batchsize', type=int, default=64, help='batch_size for training')
    parser.add_argument('--num_epochs', type=int, default=250, help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--workers', type=int, default=0, help='for data-loading')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='clip gradients at an absolute value of.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    
    # Validation
    parser.add_argument('--val_batchsize', type=int, default=1, help='batch_size for validation')
    parser.add_argument('--savepath', default="./models_ckpt/")
    
    # backbone parameters
    parser.add_argument('--network', default='segformer-mit_b1', help='define the backbone encoder to extract features')
    parser.add_argument('--encoder_dim', type=int, default=512,
                        help='the dimension of extracted features using backbone ')
    parser.add_argument('--feat_size', type=int, default=16,
                        help='define the output size of encoder to extract features')
    
    parser.add_argument('--in_height', type=int, default=256, help='Input image height for X3D')
    parser.add_argument('--in_width', type=int, default=256, help='Input image width for X3D')
    parser.add_argument('--pretrained', type=str, default='/media/xidian/xd132/CXH/change-agent/Multi_change/X3D_L.pyth', help='Path to pretrained X3D weights')
    
    # Model parameters
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in AttentionEncoder.')
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--feature_dim', type=int, default=512, help='embedding dimension')
    
    # ==================== Resume 核心逻辑 5/5: 新增参数 ====================
    parser.add_argument('--resume', action='store_true', 
                        help='自动从 savepath 中加载最新的 checkpoint_latest.pth 继续训练')
    parser.add_argument('--resume_path', default=None, type=str,
                        help='指定具体的 checkpoint 文件路径进行恢复 (优先级高于 --resume)')

    args = parser.parse_args()

    trainer = Trainer(args)
    print('Starting Epoch:', trainer.start_epoch)
    print('Total Epoches:', trainer.args.num_epochs)

    if args.train_goal == 2:
        goals = [2, 1, 0]
        # 确定起始索引
        start_idx = trainer.current_goal_index if trainer.resumed else 0
        
        for idx in range(start_idx, len(goals)):
            goal = goals[idx]
            trainer.current_goal_index = idx
            
            print_log(f'Current train_goal={goal}:\n', trainer.log)
            trainer.args.train_goal = goal
            if goal == 2:
                trainer.args.train_stage = 's1'
                trainer.args.checkpoint = None
                for epoch in range(trainer.start_epoch, trainer.args.num_epochs):
                    trainer.training(trainer.args, epoch)
                    trainer.validation(epoch)
                    if epoch - trainer.best_epoch > 50:
                        trainer.start_epoch = trainer.best_epoch + 1
                        break
                    elif epoch == trainer.args.num_epochs - 1:
                        trainer.start_epoch = trainer.best_epoch + 1
                        trainer.args.num_epochs = trainer.start_epoch + args.num_epochs
            else:
                trainer.args.train_stage = 's2'
                trainer.args.checkpoint = trainer.best_model_path
                trainer.build_model()
                for epoch in range(trainer.start_epoch, trainer.args.num_epochs):
                    trainer.training(trainer.args, epoch)
                    trainer.validation(epoch)
                    if trainer.args.train_goal == 1 and epoch - trainer.best_epoch > 50:
                        trainer.start_epoch = trainer.best_epoch + 1
                        trainer.args.num_epochs = trainer.start_epoch + trainer.args.num_epochs
                        break
                # trainer.args.num_epochs = trainer.start_epoch + trainer.args.num_epochs
    else:
        for epoch in range(trainer.start_epoch, trainer.args.num_epochs):
            trainer.training(trainer.args, epoch)
            # if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)
