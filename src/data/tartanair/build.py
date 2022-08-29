# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------


import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageFilter, ImageOps
from src.data.tartanair.tartanair_video import TartanAirVideoDataset
from src.data.tartanair.utils import ResizeFlowNP
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == "bicubic":
            return InterpolationMode.BICUBIC
        elif method == "lanczos":
            return InterpolationMode.LANCZOS
        elif method == "hamming":
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR

    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


def build_loader(args):

    # Only prepare training set for MoBY models.
    dataset_train = build_dataset(is_train=True, args=args)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
        shuffle=False,
    )

    if not args.use_val:
        return dataset_train, None, data_loader_train, None, None

    dataset_val, _ = build_dataset(is_train=False, args=args)
    # print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    if config.DATA.VAL_DATA_LOADER == "default":
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            shuffle=False,
            drop_last=False,
        )
    elif config.DATA.VAL_DATA_LOADER == "subset":

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    else:
        raise ValueError()

    return dataset_train, dataset_val, data_loader_train, data_loader_val, None


def build_dataset(is_train, args):
    if is_train:
        ann_file = args.train_ann_file
    else:
        ann_file = args.val_ann_file

    transform = build_tartanair_video_transform(is_train, args)
    dataset = TartanAirVideoDataset(
        ann_file,
        clip_len=args.num_seq,
        seq_len=args.seq_len,
        modalities=args.modalities,
        transform=transform,
        video_name_keyword=args.video_name_keyword if is_train else None,
    )

    return dataset


def build_transform(is_train, config):
    if config.AUG.SSL_AUG:
        if config.AUG.SSL_AUG_TYPE == "byol":
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            transform_1 = transforms.Compose(
                [
                    transforms.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(config.AUG.SSL_AUG_CROP, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur()], p=1.0),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
            transform_2 = transforms.Compose(
                [
                    transforms.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(config.AUG.SSL_AUG_CROP, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur()], p=0.1),
                    transforms.RandomApply([ImageOps.solarize], p=0.2),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

            transform = (transform_1, transform_2)
            return transform
        else:
            raise NotImplementedError

    if config.AUG.SSL_LINEAR_AUG:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(config.DATA.IMG_SIZE),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(config.DATA.IMG_SIZE + 32),
                    transforms.CenterCrop(config.DATA.IMG_SIZE),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        return transform

    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != "none" else None,
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
                transforms.Resize(
                    (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                    interpolation=_pil_interp(config.DATA.INTERPOLATION),
                )
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_tartanair_video_transform(is_train, args):
    if is_train:
        if args.train_transform == "TartanAirVideoTransform":
            return TartanAirVideoTransform(
                center_crop_size=args.img_crop,
                resize_size=args.img_dim,
            )
        elif args.train_transform == "TartanAirVideoTransformWithAugmentation":
            return TartanAirVideoTransformWithAugmentation(
                center_crop_size=args.img_crop,
                resize_size=args.img_dim,
                modalities=args.modalities,
            )
        elif args.train_transform == "TartanAirNoTransform":
            return TartanAirNoTransform()
        else:
            raise ValueError()
    else:
        if args.val_transform == "TartanAirVideoTransform":
            return TartanAirVideoTransform(
                center_crop_size=args.img_crop,
                resize_size=args.img_dim,
                modalities=args.modalities,
            )
        elif args.train_transform == "TartanAirNoTransform":
            return TartanAirNoTransform()
        else:
            raise ValueError()


class GaussianBlur:
    """Gaussian Blur version 2."""

    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class TartanAirVideoTransform:
    """TartanAir video transform."""

    def __init__(self, center_crop_size=448, resize_size=224, modalities=None):
        self.image_transform = transforms.Compose(
            [
                transforms.CenterCrop(
                    center_crop_size
                ),  # Since TartanAir images are 640x480, we use a center crop of 448x448 and resize it to 224x224 at default.
                transforms.Resize(resize_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )
        self.flow_transform = FlowTransform(center_crop_size=center_crop_size, resize_size=resize_size)
        self.depth_transform = DepthTransform(center_crop_size=center_crop_size, resize_size=resize_size)

    def __call__(self, item):
        transformed_item = {}
        for modality in item:
            # TODO: We need to support multiple data types in sub-transforms (e.g. CenterCrop / Resize) for code refactoring.
            if modality == "image_left":
                transform = self.image_transform
            elif modality == "flow_flow":
                transform = self.flow_transform
            elif modality == "depth_left":
                transform = self.depth_transform
            elif modality == "seg_left":
                transform = self.depth_transform
            else:
                raise NotImplementedError()

            stacked_data = torch.stack([transform(x) for x in item[modality]], dim=1)
            transformed_item[modality] = stacked_data

        return transformed_item


class TartanAirNoTransform:
    """TartanAir video transform."""

    def __init__(self):
        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __call__(self, item):
        transformed_item = {}
        for modality in item:
            # TODO: We need to support multiple data types in sub-transforms (e.g. CenterCrop / Resize) for code refactoring.
            transform = self.image_transform
            stacked_data = torch.stack([transform(x) for x in item[modality]], dim=1)
            transformed_item[modality] = stacked_data

        return transformed_item


class FlowTransform:
    """Transform for optical flow.

    Ref: https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py
    """

    def __init__(self, center_crop_size: int, resize_size: int, flow_normalizer=20.0):
        self.center_crop_size = center_crop_size
        self.resize_size = resize_size
        self.flow_normalizer = flow_normalizer
        self.resize = ResizeFlowNP(size=(resize_size, resize_size), scale_flow=True)

    def __call__(self, x):
        # H, W, _ = x.shape
        # ox, oy = int(H/2)-1, int(W/2)-1
        # delta = int(self.center_crop_size / 2)
        # x = x[ ox -delta  : ox + delta, oy-delta : oy + delta, :]

        x = self.resize(x)  # new added
        x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)  # [H,W,C] -> [C,H,W].
        # _, H, W = x.shape
        # assert self.center_crop_size < H and self.center_crop_size < W
        # x = F.center_crop(x, self.center_crop_size)
        # x = F.resize(x, self.resize_size)
        # x = float(self.resize_size) / float(self.center_crop_size) * x  # Scaling the values in the flow to match the resize function.
        # x = x / self.flow_normalizer
        return x


class DepthTransform:
    """Transform for depth."""

    def __init__(self, center_crop_size: int, resize_size: int):
        self.center_crop_size = center_crop_size
        self.resize_size = resize_size
        self.resize = ResizeFlowNP(size=(resize_size, resize_size), scale_flow=False)

    def __call__(self, x):
        H, W = x.shape
        ox, oy = int(H / 2) - 1, int(W / 2) - 1
        delta = int(self.center_crop_size / 2)
        x = x[ox - delta : ox + delta, oy - delta : oy + delta]
        x = self.resize(x)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # [H,W] -> [1,H,W].
        # _, H, W = x.shape
        # assert self.center_crop_size < H and self.center_crop_size < W
        # x = F.center_crop(x, self.center_crop_size)
        # x = F.resize(x, self.resize_size)
        x = 1.0 / x  # Inverse depth map.
        return x


class TartanAirVideoTransformWithAugmentation:
    """TartanAir video transform with augmentation.

    Currently only support color jittering and flipping.
    Ref: https://github.com/princeton-vl/RAFT/blob/224320502d66c356d88e6c712f38129e60661e80/core/utils/augmentor.py#L15
    """

    def __init__(
        self,
        center_crop_size=448,
        resize_size=224,
        modalities=None,
        do_flip=True,
        do_color_jitter=True,
    ):

        self.modalities = modalities
        # Flip augmentation params.
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # Photometric augmentation params (only color jittering).
        self.do_color_jitter = do_color_jitter
        self.color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14)
        self.asymmetric_color_aug_prob = 0.2

        self.image_transform = transforms.Compose(
            [
                transforms.CenterCrop(
                    center_crop_size
                ),  # Since TartanAir images are 640x480, we use a center crop of 448x448 and resize it to 224x224 at default.
                transforms.Resize(resize_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )
        self.flow_transform = FlowTransform(center_crop_size=center_crop_size, resize_size=resize_size)
        self.depth_transform = DepthTransform(center_crop_size=center_crop_size, resize_size=resize_size)

    # @profile
    def __call__(self, item):
        # TODO: Need a better visualization of these augmentations.
        # 1. Color jittering
        if (self.do_color_jitter) and ("image_left" in self.modalities):
            # Asymmetric.
            if np.random.rand() < self.asymmetric_color_aug_prob:
                images = [np.array(self.color_jitter(x)) for x in item["image_left"]]
            # Symmetric.
            else:
                image_stack = np.concatenate(
                    [np.array(x) for x in item["image_left"]], axis=0
                )  # Shape: [H,W,C]*D -> [D*H,W,C].
                image_stack = np.array(self.color_jitter(Image.fromarray(image_stack)), dtype=np.uint8)
                images = np.split(image_stack, len(item["image_left"]), axis=0)  # Shape: [D*H,W,C] -> [H,W,C]*D.
        else:
            images = [np.array(x) for x in item["image_left"]]

        # 2. Flipping
        if "flow_flow" in self.modalities:
            flows = item["flow_flow"]
        if "depth_left" in self.modalities:
            depths = item["depth_left"]
        if "seg_left" in self.modalities:
            segs = item["seg_left"]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                images = [x[:, ::-1] for x in images]  # Shape: [H,W,C].
                if "flow_flow" in self.modalities:
                    flows = [x[:, ::-1] * [-1.0, 1.0] for x in item["flow_flow"]]  # Shape: [H,W,2].
                if "depth_left" in self.modalities:
                    depths = [x[:, ::-1] for x in item["depth_left"]]  # Shape: [H,W,1].
                if "seg_left" in self.modalities:
                    segs = [x[:, ::-1] for x in item["seg_left"]]  # Shape: [H,W,1].
            if np.random.rand() < self.v_flip_prob:  # v-flip
                images = [x[::-1, :] for x in images]
                if "flow_flow" in self.modalities:
                    flows = [x[::-1, :] * [1.0, -1.0] for x in item["flow_flow"]]
                if "depth_left" in self.modalities:
                    depths = [x[::-1, :] for x in item["depth_left"]]
                if "seg_left" in self.modalities:
                    segs = [x[::-1, :] for x in item["seg_left"]]

        # 3. Standard transformations
        images = [Image.fromarray(x) for x in images]

        transformed_item = {}
        if "image_left" in self.modalities:
            transformed_item["image_left"] = torch.stack(
                [self.image_transform(x) for x in images], dim=1
            )  # Shape: [H,W,C]*D -> [C,H,W]*D -> [C,D,H,W].
        if "flow_flow" in self.modalities:
            transformed_item["flow_flow"] = torch.stack(
                [self.flow_transform(x) for x in flows], dim=1
            )  # Shape: [H,W,2]*D -> [2,H,W]*D -> [2,D,H,W].
        if "depth_left" in self.modalities:
            transformed_item["depth_left"] = torch.stack(
                [self.depth_transform(x) for x in depths], dim=1
            )  # Shape: [H,W,2]*D -> [2,H,W]*D -> [2,D,H,W].
        if "seg_left" in self.modalities:
            transformed_item["seg_left"] = torch.stack(
                [self.depth_transform(x) for x in segs], dim=1
            )  # Shape: [H,W,2]*D -> [2,H,W]*D -> [2,D,H,W].

        return transformed_item
