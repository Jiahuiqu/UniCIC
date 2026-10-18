import torch
from torch.utils.data import Dataset
from preprocess_data import encode
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
#import cv2 as cv
from imageio import imread
from random import *

class LEVIRCCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, list_path, split, token_folder = None, vocab_file = None, max_length = 41, allow_unk = 0, max_iters=None):
        """
        :param data_folder: folder where image files are stored
        :param list_path: folder where the file name-lists of Train/val/test.txt sets are stored
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param token_folder: folder where token files are stored
        :param vocab_file: the name of vocab file
        :param max_length: the maximum length of each caption sentence
        :param max_iters: the maximum iteration when loading the data
        :param allow_unk: whether to allow the tokens have unknow word or not
        """
        self.mean = [0.39073*255,  0.38623*255, 0.32989*255]
        self.std = [0.15329*255,  0.14628*255, 0.13648*255]
        self.list_path = list_path
        self.split = split
        self.max_length = max_length

        assert self.split in {'train', 'val', 'test'}
        self.img_ids = [i_id.strip() for i_id in open(os.path.join(list_path + split + '.txt'))]
        if vocab_file is not None:
            with open(os.path.join(list_path + vocab_file + '.json'), 'r') as f:
                self.word_vocab = json.load(f)
            self.allow_unk = allow_unk
        if not max_iters==None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]
        self.files = []
        if split =='train':
            for name in self.img_ids:
                img_fileA = os.path.join(data_folder + '/' + split +'/A/' + name.split('-')[0])
                img_fileB = img_fileA.replace('/A/', '/B/')

                imgA = imread(img_fileA)
                imgB = imread(img_fileB)
                seg_label = imread(img_fileA.replace('/A/', '/label/'))

                if '-' in name:
                    token_id = name.split('-')[-1]
                else:
                    token_id = None
                if token_folder is not None:
                    token_file = os.path.join(token_folder + name.split('.')[0] + '.txt')
                else:
                    token_file = None
                self.files.append({
                    "imgA": imgA,
                    "imgB": imgB,
                    "seg_label": seg_label,
                    "token": token_file,
                    "token_id": token_id,
                    "name": name.split('-')[0]
                })
        elif split =='val':
            for name in self.img_ids:
                img_fileA = os.path.join(data_folder + '/' + split +'/A/' + name)
                img_fileB = img_fileA.replace('/A/', '/B/')

                imgA = imread(img_fileA)
                imgB = imread(img_fileB)
                seg_label = imread(img_fileA.replace('/A/', '/label/'))

                token_id = None
                if token_folder is not None:
                    token_file = os.path.join(token_folder + name.split('.')[0] + '.txt')
                else:
                    token_file = None
                self.files.append({
                    "imgA": imgA,
                    "imgB": imgB,
                    "seg_label": seg_label,
                    "token": token_file,
                    "token_id": token_id,
                    "name": name
                })
        elif split =='test':
            for name in self.img_ids:
                img_fileA = os.path.join(data_folder + '/' + split +'/A/' + name)
                img_fileB = img_fileA.replace('/A/', '/B/')

                imgA = imread(img_fileA)
                imgB = imread(img_fileB)
                seg_label = imread(img_fileA.replace('/A/', '/label/'))

                token_id = None
                if token_folder is not None:
                    token_file = os.path.join(token_folder + name.split('.')[0] + '.txt')
                else:
                    token_file = None
                self.files.append({
                    "imgA": imgA,
                    "imgB": imgB,
                    "seg_label": seg_label,
                    "token": token_file,
                    "token_id": token_id,
                    "name": name
                })
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        name = datafiles["name"]

        imgA = datafiles["imgA"]
        imgB = datafiles["imgB"]
        seg_label = datafiles["seg_label"]

        imgA = np.asarray(imgA, np.float32)
        imgB = np.asarray(imgB, np.float32)
        imgA = imgA.transpose(2, 0, 1)
        imgB = imgB.transpose(2, 0, 1)
        seg_label = seg_label.transpose(2, 0, 1)[0]
        seg_label[seg_label==255] = 2
        seg_label[seg_label==128] = 1


        for i in range(len(self.mean)):
            imgA[i,:,:] -= self.mean[i]
            imgA[i,:,:] /= self.std[i]
            imgB[i,:,:] -= self.mean[i]
            imgB[i,:,:] /= self.std[i]      
        if datafiles["token"] is not None:
            caption = open(datafiles["token"])
            caption = caption.read()
            caption_list = json.loads(caption)

            #token = np.zeros((1, self.max_length), dtype=int)
            #j = randint(0, len(caption_list) - 1)
            #tokens_encode = encode(caption_list[j], self.word_vocab,
            #            allow_unk=self.allow_unk == 1)
            #token[0, :len(tokens_encode)] = tokens_encode
            #token_len = len(tokens_encode)

            token_all = np.zeros((len(caption_list),self.max_length),dtype=int)
            token_all_len = np.zeros((len(caption_list),1),dtype=int)
            for j, tokens in enumerate(caption_list):
                nochange_cap = ['<START>', 'the', 'scene', 'is', 'the', 'same', 'as', 'before', '<END>']
                if self.split == 'train' and nochange_cap in caption_list:
                    tokens = nochange_cap
                tokens_encode = encode(tokens, self.word_vocab,
                                    allow_unk=self.allow_unk == 1)
                token_all[j,:len(tokens_encode)] = tokens_encode
                token_all_len[j] = len(tokens_encode)
            if datafiles["token_id"] is not None:
                id = int(datafiles["token_id"])
                token = token_all[id]
                token_len = token_all_len[id].item()
            else:
                j = randint(0, len(caption_list) - 1)
                token = token_all[j]
                token_len = token_all_len[j].item()
        else:
            token_all = np.zeros(1, dtype=int)
            token = np.zeros(1, dtype=int)
            token_len = np.zeros(1, dtype=int)
            token_all_len = np.zeros(1, dtype=int)

        return imgA.copy(), imgB.copy(), seg_label.copy(), token_all.copy(), token_all_len.copy(), token.copy(), np.array(token_len), name

class SAMLEVIRCCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, list_path, split, token_folder = None, vocab_file = None, max_length = 41, allow_unk = 0, max_iters=None, sam_frame_folder=None):
        """
        :param data_folder: folder where image files are stored
        :param list_path: folder where the file name-lists of Train/val/test.txt sets are stored
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param token_folder: folder where token files are stored
        :param vocab_file: the name of vocab file
        :param max_length: the maximum length of each caption sentence
        :param max_iters: the maximum iteration when loading the data
        :param allow_unk: whether to allow the tokens have unknow word or not
        :param sam_frame_folder: folder where preprocessed SAM frames (.pt) are stored
        """
        self.mean = [0.39073*255,  0.38623*255, 0.32989*255]
        self.std = [0.15329*255,  0.14628*255, 0.13648*255]
        self.list_path = list_path
        self.split = split
        self.max_length = max_length
        self.sam_frame_folder = sam_frame_folder # [MODIFIED] Store SAM frame path

        assert self.split in {'train', 'val', 'test'}
        self.img_ids = [i_id.strip() for i_id in open(os.path.join(list_path + split + '.txt'))]
        if vocab_file is not None:
            with open(os.path.join(list_path + vocab_file + '.json'), 'r') as f:
                self.word_vocab = json.load(f)
            self.allow_unk = allow_unk
        if not max_iters==None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]
        
        self.files = []
        
        # Helper to get SAM frame path
        def get_sam_path(name):
            if self.sam_frame_folder is None:
                return None
            # Name in txt might have suffix like .png, remove it and add .pt
            base_name = name.split('.')[0] 
            # Handle case where name might contain '-' for token ID (though usually filename is clean in txt)
            if '-' in base_name:
                 base_name = base_name.split('-')[0]
            
            pt_name = base_name + '.pt'
            path = os.path.join(self.sam_frame_folder, self.split, pt_name)
            if os.path.exists(path):
                return path
            return None

        if split =='train':
            for name in self.img_ids:
                # name in train.txt is usually just filename.png
                img_fileA = os.path.join(data_folder + '/' + split +'/A/' + name.split('-')[0])
                img_fileB = img_fileA.replace('/A/', '/B/')

                # Check if files exist before appending to avoid crash later
                if not os.path.exists(img_fileA) or not os.path.exists(img_fileB):
                    continue

                imgA = imread(img_fileA)
                imgB = imread(img_fileB)
                seg_label = imread(img_fileA.replace('/A/', '/label/'))

                if '-' in name:
                    token_id = name.split('-')[-1]
                else:
                    token_id = None
                if token_folder is not None:
                    token_file = os.path.join(token_folder + name.split('.')[0] + '.txt')
                else:
                    token_file = None
                
                sam_frame_path = get_sam_path(name) # [MODIFIED] Get SAM path

                self.files.append({
                    "imgA": imgA,
                    "imgB": imgB,
                    "seg_label": seg_label,
                    "token": token_file,
                    "token_id": token_id,
                    "name": name.split('-')[0],
                    "sam_frame": sam_frame_path # [MODIFIED] Add to dict
                })
        elif split =='val':
            for name in self.img_ids:
                img_fileA = os.path.join(data_folder + '/' + split +'/A/' + name)
                img_fileB = img_fileA.replace('/A/', '/B/')

                if not os.path.exists(img_fileA) or not os.path.exists(img_fileB):
                    continue

                imgA = imread(img_fileA)
                imgB = imread(img_fileB)
                seg_label = imread(img_fileA.replace('/A/', '/label/'))

                token_id = None
                if token_folder is not None:
                    token_file = os.path.join(token_folder + name.split('.')[0] + '.txt')
                else:
                    token_file = None
                
                sam_frame_path = get_sam_path(name) # [MODIFIED] Get SAM path

                self.files.append({
                    "imgA": imgA,
                    "imgB": imgB,
                    "seg_label": seg_label,
                    "token": token_file,
                    "token_id": token_id,
                    "name": name,
                    "sam_frame": sam_frame_path # [MODIFIED] Add to dict
                })
        elif split =='test':
            for name in self.img_ids:
                img_fileA = os.path.join(data_folder + '/' + split +'/A/' + name)
                img_fileB = img_fileA.replace('/A/', '/B/')

                if not os.path.exists(img_fileA) or not os.path.exists(img_fileB):
                    continue

                imgA = imread(img_fileA)
                imgB = imread(img_fileB)
                seg_label = imread(img_fileA.replace('/A/', '/label/'))

                token_id = None
                if token_folder is not None:
                    token_file = os.path.join(token_folder + name.split('.')[0] + '.txt')
                else:
                    token_file = None
                
                sam_frame_path = get_sam_path(name) # [MODIFIED] Get SAM path

                self.files.append({
                    "imgA": imgA,
                    "imgB": imgB,
                    "seg_label": seg_label,
                    "token": token_file,
                    "token_id": token_id,
                    "name": name,
                    "sam_frame": sam_frame_path # [MODIFIED] Add to dict
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        name = datafiles["name"]

        imgA = datafiles["imgA"]
        imgB = datafiles["imgB"]
        seg_label = datafiles["seg_label"]

        imgA = np.asarray(imgA, np.float32)
        imgB = np.asarray(imgB, np.float32)
        imgA = imgA.transpose(2, 0, 1)
        imgB = imgB.transpose(2, 0, 1)
        seg_label = seg_label.transpose(2, 0, 1)[0]
        seg_label[seg_label==255] = 2
        seg_label[seg_label==128] = 1


        for i in range(len(self.mean)):
            imgA[i,:,:] -= self.mean[i]
            imgA[i,:,:] /= self.std[i]
            imgB[i,:,:] -= self.mean[i]
            imgB[i,:,:] /= self.std[i]      
        
        # [MODIFIED START] Load SAM frame ingredients (式(14) 的区域项 + 背景项)
        # 缓存文件为 [2, C, H, W]：第 0 份区域项 F_reg，第 1 份背景项 S_proj。
        # α 的混合在训练期由 SAMFrameFusion 完成，这里只负责把原料喂进去。
        sam_region_tensor = torch.zeros((3, 256, 256))
        sam_bg_tensor = torch.zeros((3, 256, 256))
        if datafiles["sam_frame"] is not None and os.path.exists(datafiles["sam_frame"]):
            try:
                blob = torch.load(datafiles["sam_frame"], map_location='cpu')
                if not (torch.is_tensor(blob) and blob.dim() == 4 and blob.shape[0] == 2):
                    raise ValueError(
                        f'unexpected cached shape {tuple(getattr(blob, "shape", ()))}; '
                        'expected [2, C, H, W] (region term + background term). '
                        'Re-run preprocess_sam_frames.py to regenerate the cache.'
                    )
                # Ensure it's float32
                sam_region_tensor = blob[0].float()
                sam_bg_tensor = blob[1].float()
            except Exception as e:
                print(f"Warning: Could not load SAM frame {datafiles['sam_frame']}: {e}")
                # Fallback to zeros if loading fails: 两份原料全零 -> 感知帧退化为纯随机基帧
                sam_region_tensor = torch.zeros((3, 256, 256))
                sam_bg_tensor = torch.zeros((3, 256, 256))
# [MODIFIED END]

        if datafiles["token"] is not None:
            caption = open(datafiles["token"])
            caption = caption.read()
            caption_list = json.loads(caption)

            token_all = np.zeros((len(caption_list),self.max_length),dtype=int)
            token_all_len = np.zeros((len(caption_list),1),dtype=int)
            for j, tokens in enumerate(caption_list):
                nochange_cap = ['<START>', 'the', 'scene', 'is', 'the', 'same', 'as', 'before', '<END>']
                if self.split == 'train' and nochange_cap in caption_list:
                    tokens = nochange_cap
                tokens_encode = encode(tokens, self.word_vocab,
                                    allow_unk=self.allow_unk == 1)
                token_all[j,:len(tokens_encode)] = tokens_encode
                token_all_len[j] = len(tokens_encode)
            if datafiles["token_id"] is not None:
                id = int(datafiles["token_id"])
                token = token_all[id]
                token_len = token_all_len[id].item()
            else:
                j = randint(0, len(caption_list) - 1)
                token = token_all[j]
                token_len = token_all_len[j].item()
        else:
            token_all = np.zeros(1, dtype=int)
            token = np.zeros(1, dtype=int)
            token_len = np.zeros(1, dtype=int)
            token_all_len = np.zeros(1, dtype=int)

        # [MODIFIED] Return the SAM frame ingredients as well
        # Note: You need to update your train.py collate_fn or unpacking logic to handle these new return values
        return (imgA.copy(), imgB.copy(), seg_label.copy(), token_all.copy(), token_all_len.copy(),
                token.copy(), np.array(token_len), name, sam_region_tensor, sam_bg_tensor)