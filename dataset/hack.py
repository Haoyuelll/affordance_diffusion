from PIL import Image
import numpy as np
import os.path as osp
from random import randint, uniform, random
import cv2
import PIL
from PIL import Image, ImageFilter
import pandas as pd
import json
import torch as th
import torchvision.transforms.functional as F
from torchvision import transforms as T
from torchvision.transforms import Compose, RandomApply, ToPILImage, ToTensor
from torch.utils.data import Dataset
from utils.glide_utils import get_uncond_tokens_mask
from utils.train_utils import pil_image_to_norm_tensor

def random_resized_crop(image, shape, resize_ratio=1.0, return_T=False):
    """
    Randomly resize and crop an image to a given size.

    Args:
        image (PIL.Image): The image to be resized and cropped.
        shape (tuple): The desired output shape.
        resize_ratio (float): The ratio to resize the image.
    """
    image_transform = T.RandomResizedCrop(shape, scale=(resize_ratio, 1.0), ratio=(1.0, 1.0))
    if not return_T:
        return image_transform(image)
    return image_transform


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

        
class HO3Pairs(Dataset):
    def __init__(
        self,
        folder="HOI4D_glide",
        split='train_handcrop.csv',
        side_x=64,
        side_y=64,
        resize_ratio=0.75,
        shuffle=False,
        tokenizer=None,
        mask_mode='lollipop',
        use_flip=False,
        is_train=True,
        cfg={},
        data_cfg={},
    ):
        super().__init__()
        self.data_cfg = data_cfg
        self.cfg = cfg
        self.data_dir = folder
        self.split = split
        self.image_dir = osp.join(folder, 'glide_hoi/{}.png')
        self.obj_dir = osp.join(folder, 'glide_obj/{}.png')
        self.sub_dir = osp.join(folder, '%s/{}.png' % cfg.sub_dir)
        self.mask_dir = osp.join(folder, 'det_mask/{}.png')
        self.hand_dir = osp.join(folder, 'det_hand/{}.json')
        self.hoi_box_dir = osp.join(folder, 'hoi_box/{}.json')

        self.mask_mode = mask_mode
        self.use_flip = use_flip
        self.is_train = is_train

        self.transform = Compose(
            [RandomApply([GaussianBlur([.1, 2.])], p=cfg.jitter_p)]
        )
        # self.image_files = list(iou_dict.keys())
        self.image_files = []
        if '.csv' in split:
            df = pd.read_csv(split)
            for i, data in df.iterrows():
                self.image_files.append(
                    '{}_frame{:04d}'.format(
                        data['vid_index'].replace('/', '_'), data['frame_number']))
        else:
            self.image_files = [index.strip() for index in open(split)]

        self.resize_ratio = resize_ratio

        self.shuffle = shuffle
        self.prefix = folder
        self.side_x = side_x
        self.side_y = side_y
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.image_files)

    def random_sample(self):
        return self.__getitem__(np.random.randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)
    
    def load_hand_box(self, ind):
        obj = json.load(open(self.hand_dir.format(self.image_files[ind])))
        box_list = []
        def xywh2xyxy(xywh):
            x1, y1, w, h = xywh
            return [x1, y1, x1+w, y1+h]

        if self.data_cfg.get('xywh', True):
            f_box = xywh2xyxy
        else:
            f_box = lambda x: x
        # to fix hand box file bug
        if isinstance(obj['hand_bbox_list'][0], list):
            obj['hand_bbox_list'] = obj['hand_bbox_list'][0]
        for hand_box in obj['hand_bbox_list']:
            if 'right_hand' in hand_box:
                box_list.append(f_box(hand_box['right_hand']))
            if 'left_hand' in hand_box:
                box_list.append(f_box(hand_box['left_hand']))
        return np.array(box_list)

    def __getitem__(self, ind):
        image_file = self.image_dir.format(self.image_files[ind])

        # null text
        tokens, mask = get_uncond_tokens_mask(self.tokenizer)
        text = ''

        try:
            original_pil_image = PIL.Image.open(image_file).convert("RGB")
        except (OSError, ValueError) as e:
            print(f"An exception occurred trying to load file {image_file}.", self.split)
            print(f"Skipping index {ind}")
            print(e)
            return self.skip_sample(ind)

        # get corresponding object-only image
        try:
            obj_file = self.get_obj_file(ind)
            mask_file = self.mask_dir.format(self.image_files[ind])

            original_pil_obj = PIL.Image.open(obj_file).convert("RGB")
            original_pil_mask = PIL.Image.open(mask_file).convert("RGB")
            original_pil_obj = self.preprocess_obj(original_pil_obj)
        except (FileNotFoundError, OSError, ValueError, cv2.error) as e:
            print(f"An exception occurred trying to load file {obj_file, mask_file}.", self.split)
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        trans = random_resized_crop(original_pil_image, (self.side_x, self.side_y), resize_ratio=self.resize_ratio, return_T=True)
        i, j, h, w = trans.get_params(original_pil_image, trans.scale, trans.ratio)
        base_pil_image = F.resized_crop(original_pil_image, i, j, h, w, trans.size, trans.interpolation)
        base_pil_obj = F.resized_crop(original_pil_obj, i, j, h, w, trans.size, trans.interpolation)
        base_pil_mask = F.resized_crop(original_pil_mask, i, j, h, w, trans.size, trans.interpolation)
        
        bboxes = None

        if random() > 0.5 and self.use_flip:
            base_pil_image = F.hflip(base_pil_image)
            base_pil_obj = F.hflip(base_pil_obj)
            base_pil_mask = F.hflip(base_pil_mask)

        try:
            base_pil_mask, mask_param = self.preprocess_mask(
                base_pil_mask, self.is_train, bboxes=bboxes, **self.cfg.jitter, iou_th=self.data_cfg.get('iou', 0.5), ind=ind)
        except cv2.error as e:
            print(f"An exception occurred preprocessing mask {mask_file}.", self.split)
            print(e)
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        base_tensor = pil_image_to_norm_tensor(base_pil_image)
        base_obj = pil_image_to_norm_tensor(base_pil_obj)
        base_mask = (th.FloatTensor(np.asarray(base_pil_mask)> 0))[None]
        return th.LongTensor(tokens), th.BoolTensor(mask), base_tensor, \
            base_obj, base_mask, mask_param.astype(np.float32).reshape(-1), text