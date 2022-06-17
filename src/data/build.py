# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

import os
import torch
import numpy as np
from PIL import ImageFilter, ImageOps
import torch.utils.data
import torch.distributed as dist
from torchvision import transforms
from torchvision.transforms import functional as F
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
#from timm.data.transforms import _str_to_pil_interpolation 
from timm.data.transforms import _pil_interp
from PIL import Image

from .cached_image_folder import CachedImageFolder
from .custom_image_folder import CustomImageFolder
from .tartanair_video import TartanAirVideoDataset
from .samplers import SubsetRandomSampler, SubsetSampler
#from utils.utils import ResizeFlowNP

import sys
#sys.path.append('../')
from utils.utils import ResizeFlowNP



def build_loader(args, data_type):
    #model_type = config.MODEL.TYPE

    # Only prepare training set for MoBY models.
    dataset_train = build_dataset(is_train=True, args=args, data_type=data_type)
    #print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")

    #num_tasks = dist.get_world_size()
    #global_rank = dist.get_rank()
    #sampler_train = torch.utils.data.DistributedSampler(
    #    dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    #)

    #sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
        drop_last=True,
        shuffle=False
    )

    if not args.use_val:
        return dataset_train, None, data_loader_train, None, None

    dataset_val, _ = build_dataset(is_train=False, config=config)
    #print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    if config.DATA.VAL_DATA_LOADER == 'default':
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=args.pin_memory,
            shuffle=False,
            drop_last=False
        )
    elif config.DATA.VAL_DATA_LOADER == 'subset':
        #indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
        #sampler_val = torch.utils.data.RandomSampler(dataset_val)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=args.pin_memory,
            shuffle=False,
            drop_last=False
        )

    else:
        raise ValueError()

    return dataset_train, dataset_val, data_loader_train, data_loader_val, None


def build_dataset(is_train, args, data_type):
    #if config.DATA.DATASET == 'imagenet':
    #    transform = build_transform(is_train, config)
    #    prefix = 'train' if is_train else 'val'
    #    if config.DATA.ZIP_MODE:
    #        ann_file = prefix + "_map.txt"
    #        prefix = prefix + ".zip@/"
    #        dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
    #                                    cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
    #    else:
    #        # ToDo: test custom_image_folder
    #        root = os.path.join(config.DATA.DATA_PATH, prefix)
    #        dataset = CustomImageFolder(root, transform=transform)
    #    nb_classes = 1000
    #elif config.DATA.DATASET == 'tartanair':
    #assert config.DATA.ZIP_MODE
    if is_train:
        ann_file = args.train_ann_file
        #ann_type = config.DATA.TARTANAIR.TRAIN_ANN_TYPE
    else:
        ann_file = args.val_ann_file
        #ann_type = config.DATA.TARTANAIR.VAL_ANN_TYPE

    #if ann_type == 'image': # Image annotations.
    #    transform = build_transform(is_train, config)
    #    prefix = ''
    #    dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
    #                                cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
    #elif ann_type == 'video': # Video annotations.
    transform = build_tartanair_video_transform(is_train, args, data_type)
    dataset = TartanAirVideoDataset(ann_file, 
                                        clip_len=args.num_seq, 
                                        seq_len=args.seq_len,
                                        #data_types=config.DATA.TARTANAIR.VIDEO.DATA_TYPES, 
                                        data_types=data_type,
                                        transform=transform,
                                        video_name_keyword=args.video_name_keyword if is_train else None)   
    #else:
    #    raise ValueError()

    #    nb_classes = 1
    #else:
    #    raise NotImplementedError("We only support ImageNet and TartanAir Now.")

    return dataset


def build_transform(is_train, config):
    if config.AUG.SSL_AUG:
        if config.AUG.SSL_AUG_TYPE == 'byol':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            transform_1 = transforms.Compose([
                transforms.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(config.AUG.SSL_AUG_CROP, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=1.0),
                transforms.ToTensor(),
                normalize,
            ])
            transform_2 = transforms.Compose([
                transforms.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(config.AUG.SSL_AUG_CROP, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=0.1),
                transforms.RandomApply([ImageOps.solarize], p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
            
            transform = (transform_1, transform_2)
            return transform
        else:
            raise NotImplementedError
    
    if config.AUG.SSL_LINEAR_AUG:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        if is_train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(config.DATA.IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(config.DATA.IMG_SIZE + 32),
                transforms.CenterCrop(config.DATA.IMG_SIZE),
                transforms.ToTensor(),
                normalize,
            ])
        return transform
    
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_tartanair_video_transform(is_train, args, data_type):
    if is_train:
        if args.train_transform == 'TartanAirVideoTransform':
            return TartanAirVideoTransform(
                center_crop_size=args.img_crop,
                resize_size=args.img_dim,
            )
        elif args.train_transform_aug == 'TartanAirVideoTransformWithAugmentation':
            return TartanAirVideoTransformWithAugmentation(
                center_crop_size=args.img_crop,
                resize_size=args.img_dim,
                modality=data_type
            )
        else:
            raise ValueError()
    else:
        if args.val_transform == 'TartanAirVideoTransform':
            return TartanAirVideoTransform(
                center_crop_size=args.img_crop,
                resize_size=args.img_dim,
                modality=args.modality
            )
        else:
            raise ValueError()

class GaussianBlur(object):
    """Gaussian Blur version 2"""

    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class TartanAirVideoTransform(object):
    """TartanAir video transform"""
    def __init__(self, center_crop_size=448, resize_size=224, modality=None):
        self.image_transform = transforms.Compose([
            transforms.CenterCrop(center_crop_size),  # Since TartanAir images are 640x480, we use a center crop of 448x448 and resize it to 224x224 at default.
            transforms.Resize(resize_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
        self.flow_transform = FlowTransform(center_crop_size=center_crop_size, resize_size=resize_size)
        self.depth_transform = DepthTransform(center_crop_size=center_crop_size, resize_size=resize_size)

    def __call__(self, item):
        transformed_item = {}
        for data_type in item:
            # TODO: We need to support multiple data types in sub-transforms (e.g. CenterCrop / Resize) for code refactoring.
            if data_type == 'image_left':
                transform = self.image_transform
            elif data_type == 'flow_flow':
                transform = self.flow_transform
            elif data_type == 'depth_left':
                transform = self.depth_transform
            elif data_type == 'seg_left':
                transform = self.depth_transform
            else:
                raise NotImplementedError()

            stacked_data = torch.stack([transform(x) for x in item[data_type]], dim=1)
            transformed_item[data_type] = stacked_data

        return transformed_item


class FlowTransform(object):
    """ 
    Transform for optical flow.
    Ref: https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py
    """
    def __init__(self, center_crop_size: int, resize_size: int, flow_normalizer=20.0):
        self.center_crop_size = center_crop_size
        self.resize_size = resize_size
        self.flow_normalizer = flow_normalizer
        self.resize = ResizeFlowNP(size=(resize_size, resize_size), scale_flow=False)
    
    def __call__(self, x):
        H, W, _ = x.shape
        ox, oy = int(H/2)-1, int(W/2)-1 
        delta = int(self.center_crop_size / 2)
        x = x[ ox -delta  : ox + delta, oy-delta : oy + delta, :]

        x = self.resize(x) #new added
        x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)       # [H,W,C] -> [C,H,W].
        #_, H, W = x.shape
        #assert self.center_crop_size < H and self.center_crop_size < W
        #x = F.center_crop(x, self.center_crop_size)
        #x = F.resize(x, self.resize_size)
        x = float(self.resize_size) / float(self.center_crop_size) * x  # Scaling the values in the flow to match the resize function.
        x = x / self.flow_normalizer
        return x


class DepthTransform(object):
    """ 
    Transform for depth.
    """
    def __init__(self, center_crop_size: int, resize_size: int):
        self.center_crop_size = center_crop_size
        self.resize_size = resize_size
        self.resize = ResizeFlowNP(size=(resize_size, resize_size), scale_flow=False)
        
    def __call__(self, x):
        H, W = x.shape
        ox, oy = int(H/2)-1, int(W/2)-1 
        delta = int(self.center_crop_size / 2)
        x = x[ ox -delta  : ox + delta, oy-delta : oy + delta]
        x = self.resize(x)        
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)      # [H,W] -> [1,H,W].
        #_, H, W = x.shape
        #assert self.center_crop_size < H and self.center_crop_size < W
        #x = F.center_crop(x, self.center_crop_size)
        #x = F.resize(x, self.resize_size)
        x = 1.0 / x   # Inverse depth map.
        return x


class TartanAirVideoTransformWithAugmentation(object):
    """
    TartanAir video transform with augmentation.
    Currently only support color jittering and flipping.
    Ref: https://github.com/princeton-vl/RAFT/blob/224320502d66c356d88e6c712f38129e60661e80/core/utils/augmentor.py#L15
    """
    def __init__(self, center_crop_size=448, resize_size=224, modality=None, do_flip=True, do_color_jitter=True):

        self.modality = modality
        # Flip augmentation params.
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # Photometric augmentation params (only color jittering).
        self.do_color_jitter = do_color_jitter
        self.color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2

        self.image_transform = transforms.Compose([
            transforms.CenterCrop(center_crop_size),  # Since TartanAir images are 640x480, we use a center crop of 448x448 and resize it to 224x224 at default.
            transforms.Resize(resize_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
        self.flow_transform = FlowTransform(center_crop_size=center_crop_size, resize_size=resize_size)
        self.depth_transform = DepthTransform(center_crop_size=center_crop_size, resize_size=resize_size)

    def __call__(self, item):
        # TODO: Need a better visualization of these augmentations.
        # 1. Color jittering
        if self.do_color_jitter:
            # Asymmetric.
            if np.random.rand() < self.asymmetric_color_aug_prob:
                images = [np.array(self.color_jitter(x)) for x in item['image_left']]
            # Symmetric.
            else:
                image_stack = np.concatenate([np.array(x) for x in item['image_left']], axis=0)         # Shape: [H,W,C]*D -> [D*H,W,C].
                image_stack = np.array(self.color_jitter(Image.fromarray(image_stack)), dtype=np.uint8)
                images = np.split(image_stack, len(item['image_left']), axis=0)                         # Shape: [D*H,W,C] -> [H,W,C]*D.
        else:
            images = [np.array(x) for x in item['image_left']]

        # 2. Flipping
        if 'flow_flow' in self.modality:
            flows = item['flow_flow']
        if 'depth_left' in self.modality:
            depths = item['depth_left']
        if 'seg_left' in self.modality:
            segs = item['seg_left']


        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                images = [x[:, ::-1] for x in images]                                                   # Shape: [H,W,C].
                if 'flow_flow' in self.modality:
                    flows = [x[:, ::-1] * [-1.0, 1.0] for x in item['flow_flow']]                           # Shape: [H,W,2].
                if 'depth_left' in self.modality:
                    depths = [x[:, ::-1] for x in item['depth_left']]                           # Shape: [H,W,1].
                if 'seg_left' in self.modality:
                    segs = [x[:, ::-1] for x in item['seg_left']]                           # Shape: [H,W,1].
            if np.random.rand() < self.v_flip_prob: # v-flip
                images = [x[::-1, :] for x in images]
                if 'flow_flow' in self.modality:
                    flows = [x[::-1, :] * [1.0, -1.0] for x in item['flow_flow']]            
                if 'depth_left' in self.modality:
                    depths = [x[::-1, :] for x in item['depth_left']]            
                if 'seg_left' in self.modality:
                    segs = [x[::-1, :] for x in item['seg_left']]            
                

        # 3. Standard transformations
        images = [Image.fromarray(x) for x in images]

        transformed_item = {}
        transformed_item['image_left'] = torch.stack([self.image_transform(x) for x in images], dim=1)  # Shape: [H,W,C]*D -> [C,H,W]*D -> [C,D,H,W].
        if 'flow_flow' in self.modality:
            transformed_item['flow_flow'] = torch.stack([self.flow_transform(x) for x in flows], dim=1)     # Shape: [H,W,2]*D -> [2,H,W]*D -> [2,D,H,W].
        if 'depth_left' in self.modality:
            transformed_item['depth_left'] = torch.stack([self.depth_transform(x) for x in depths], dim=1)     # Shape: [H,W,2]*D -> [2,H,W]*D -> [2,D,H,W].
        if 'seg_left' in self.modality:
            transformed_item['seg_left'] = torch.stack([self.depth_transform(x) for x in segs], dim=1)     # Shape: [H,W,2]*D -> [2,H,W]*D -> [2,D,H,W].
        
        return transformed_item
