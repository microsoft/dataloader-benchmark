"""Implement utility functions and classes for dataloader. Mainly transforms for multiple modalities.
"""

import sys
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def instantiate_modal_class(args: Dict[str, Any]) -> Any:
    """Instantiates a class [defined in this module] with the given args that contains class name and init args.
    Args:
        init: Dict of the form {"modal":...,"$attributes":...}.
    Returns:
        The instantiated class object.
    """
    args_class = getattr(sys.modules[__name__], args["modal"])  # get class object from this module by its class name
    return args_class(**args["kwargs"])


class InputConfig:
    def __init__(self) -> None:
        self.inputs = []

    def add_input(self, input: Dict[str, Any]) -> None:
        self.inputs.append(instantiate_modal_class(input))

    def pre_transform(self, x: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        res = {}
        for input in self.inputs:
            key = input.name
            res[key] = input.pre_transform(x[key])
        return res

    def train_transform(self, x: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        res = {}
        for input in self.inputs:
            key = input.name
            res[key] = input.train_transform(x[key])
        return res

    def val_transform(self, x: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        res = {}
        for input in self.inputs:
            key = input.name
            res[key] = input.val_transform(x[key])
        return res


class Modality(ABC):
    def pre_transform(self, x: np.ndarray) -> np.ndarray:
        """implement transform applied to raw data when data is being stored as
        numpy arrary.

        Args:
            x (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        pass

    def train_transform(self, x: np.ndarray) -> torch.Tensor:
        """Implement transform applied to data being used as training data when
        dataloader reads it from datapipe.

        Args:
            x (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        pass

    def val_transform(self, x: np.ndarray) -> torch.Tensor:
        """Implement transform applied to data being used as validation data when
        dataloader reads it from datapipe.
        """
        pass


@dataclass
class RGBImage(Modality):
    """

    Args:
        Modality (ABC): RGB images
    """

    name: str
    img_crop: int
    img_dim: int

    def __post_init__(self):
        # transforms.Compose is not supported by torch jit script.
        # see: https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py
        # center crop and resize: If the image is torch Tensor, it is expected to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
        # normalize: essential code is tensor.sub_(mean).div_(std), so it should be fine with 'sequence' (batched) data.
        train_trans = torch.nn.Sequential(
            transforms.CenterCrop(
                self.img_crop
            ),  # Since TartanAir images are 640x480, we use a center crop of 448x448 and resize it to 224x224 at default.
            transforms.Resize([self.img_dim, self.img_dim]),  # transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        )
        # self.train_trans = torch.jit.script(train_trans)
        self.train_trans = train_trans

    def pre_transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def train_transform(self, x: np.ndarray) -> torch.Tensor:
        """Apply transform to batch of images represented by NP.NDARRAY, not pil image!

        Args:
            x (np.ndarray): batch of rgb image represented by np.ndarry with shape [seq_len, h, w, c]

        Returns:
            torch.Tensor:
        """
        # see: https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py
        tt = torch.from_numpy(x).permute(
            0, 3, 1, 2
        )  # [t, h, w, c] -> [t, c, h, w] required by torch transform applied to tensor.
        # todo: check if this is alright
        tt = tt.float().div(255)
        tt = self.train_trans(tt)  # [t, c, h, w]
        return tt

    def val_transform(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x)


@dataclass
class OpticalFlow(Modality):
    """resize permute

    Args:
        Modality (ABC): optical flow
    """

    name: str
    img_dim: int
    flow_normalizer: float

    def pre_transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def train_transform(self, x: np.ndarray) -> torch.Tensor:
        """Apply transform to batch of optical flow image

        Args:
            x (np.ndarray): batch of optical flow image

        Returns:
            torch.Tensor: _description_
        """
        tt = torch.from_numpy(x).permute(0, 3, 1, 2)  # [t, h, w, c] -> [t, c, h, w]
        tt = flow_trans(tt, self.img_dim)
        return tt

    def val_transform(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x)


@dataclass
class Depth(Modality):
    """more complicated stuff

    Args:
        Modality (ABC): Depth modality
    """

    name: str
    img_crop: int
    img_dim: int

    def __post_init__(self):
        train_trans = torch.nn.Sequential(
            transforms.CenterCrop(size=self.img_crop),
            transforms.Resize([self.img_dim, self.img_dim], interpolation=InterpolationMode.BILINEAR),
        )
        self.train_trans = train_trans

    def pre_transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def train_transform(self, x: np.ndarray) -> torch.Tensor:
        """Apply transform the batch of depth image tensor of dim [t, h, w, c].
        Each depth image is two dimensional float array

        Args:
            x (np.ndarray): _description_

        Returns:
            torch.Tensor: transformed image tensor of size [t, c, h, w]
        """
        tt = torch.from_numpy(x)
        tt = self.train_trans(tt)
        tt = 1.0 / tt
        return tt

    def val_transform(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x)


def flow_trans(x: torch.Tensor, img_dim: int) -> torch.Tensor:
    """Apply transformation to flow

    Args:
        x (torch.Tensor): 4d tensor with shape [t, c, h, w]
        img_dim (int): dimension of the resized image

    Returns:
        torch.Tensor: _description_
    """
    h, w = x.size(2), x.size(3)
    # InterpolationMode.BILINEAR produces different results to cv2.resize(interpolation.linear).
    # see https://stackoverflow.com/questions/63519965/torch-transform-resize-vs-cv2-resize
    x = transforms.Resize([img_dim, img_dim], interpolation=InterpolationMode.BILINEAR)(x)
    x[:, 0, :, :] = x[:, 0, :, :] * (float(img_dim) / float(w))
    x[:, 1, :, :] = x[:, 1, :, :] * (float(img_dim) / float(h))
    return x
