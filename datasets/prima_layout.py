import os
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data

def colorize_mask(mask):
    palette = [0,0,0, 64,128,64, 128,0,192]
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

class PRIMA(data.Dataset):
    def __init__(self, mode, joint_transform=None, sliding_crop=None, transform=None, target_transform=None):
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform

        self.img_base_dir = "/data/weihong_ma/dataset/PRImA_Layout_Analysis_Dataset/Images"
        if mode=='train':
            txt_path = '/data/weihong_ma/experiment/exp1_pytorch-semantic-segmentation/train/PRIMA-fcn/train.txt'
            self.files = open(txt_path).read().splitlines()
        elif mode=='val':
            txt_path = '/data/weihong_ma/experiment/exp1_pytorch-semantic-segmentation/train/PRIMA-fcn/test.txt'
            self.files = open(txt_path).read().splitlines()

    def __getitem__(self, index):
        img_path = os.path.join(self.img_base_dir, self.files[index])
        mask_path = img_path.replace("Images", "3_cls_mask").replace("tif", "png")

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)

        if self.sliding_crop is not None:
            img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)

            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            if self.target_transform is not None:
                mask_slices = [self.target_transform(e) for e in mask_slices]
            img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
            return img, mask, torch.LongTensor(slices_info)
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            return img, mask

    def __len__(self):
        return len(self.files)