"""Implement utility functions and classes for dataloader.

Mainly transforms for multiple modalities.
"""

import sys
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def instantiate_modal_class(args: Dict[str, Any]) -> Any:
    """Instantiates a class [defined in this module] with the given args that contains class name
    and init args.

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
    """An abstract class for each input, not inherited by any class because torch.jit currently does not
    support inheritence.

    Args:
        ABC (_type_): _description_
    """
    def pre_transform(self, x: np.ndarray) -> np.ndarray:
        """implement transform applied to raw data when data is being stored as numpy arrary.

        Args:
            x (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        pass

    def train_transform(self, x: np.ndarray) -> torch.Tensor:
        """Implement transform applied to data being used as training data when dataloader reads it
        from datapipe.

        Original input argument is np.ndarray, we have a warpper args_nparray2tensor that transforms the input to torch for train/val transforms so as to use torch jit.

        Args:
            x (np.ndarray): _description_

        Returns:
            torch.Tensor: _description_
        """
        pass

    def val_transform(self, x: np.ndarray) -> torch.Tensor:
        """Implement transform applied to data being used as validation data when
        dataloader reads it from datapipe.

        Args:
            x (np.ndarray): _description_

        Returns:
            torch.Tensor: _description_
        """
        pass

class RGBImage(Modality):
    """

    Args:
        Modality (): RGB images
    """

    def __init__(self, name:str, img_crop:int, img_dim:int):
        self.name:str = name
        # transforms.Compose is not supported by torch jit script.
        # see: https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py
        # center crop and resize: If the image is torch Tensor, it is expected to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
        # normalize: essential code is tensor.sub_(mean).div_(std), so it should be fine with 'sequence' (batched) data.
        # transforms.Compose is not supported by torch jit script.
        train_trans = torch.nn.Sequential(
            transforms.CenterCrop(
                img_crop
            ),  # Since TartanAir images are 640x480, we use a center crop of 448x448 and resize it to 224x224 at default.
            transforms.Resize([img_dim, img_dim]),  # transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        )
        self.train_trans = train_trans
        # torch.jit.script(train_trans) will slow the program 

    def pre_transform(self, x: np.ndarray) -> np.ndarray:
        # torch.jit cannot deal with numpy
        return x

    def train_transform(self, x: np.ndarray) -> torch.Tensor:
        """Apply transform to batch of images represented by torch.Tensor, not pil image!

        Args:
            x (np.ndarray): batch of rgb image represented by np.ndarry with shape [seq_len, h, w, c]

        Returns:
            torch.Tensor:
        """
        @torch.jit.script
        def jit_trans(trans:torch.nn.Module, tt: torch.Tensor):
            # see: https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py
            tt = tt.permute(
                0, 3, 1, 2
            )  # [t, h, w, c] -> [t, c, h, w] required by torch transform applied to tensor.
            tt = tt.float().div(255)
            tt = trans(tt)  # [t, c, h, w]
            return tt

        return jit_trans(self.train_trans, torch.from_numpy(x))

    def val_transform(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x)


class OpticalFlow(Modality):
    """resize permute.

    Args:
        Modality (): optical flow
    """

    def __init__(self, name:str, img_dim:int, flow_normalizer:float):
        self.name:str = name
        self.img_dim :int = img_dim
        self.flow_normalizer :float = flow_normalizer
        train_trans = transforms.Resize([img_dim, img_dim], interpolation=InterpolationMode.BILINEAR)
        self.train_trans = train_trans

    def pre_transform(self, x: np.ndarray) -> np.ndarray:
        # torch.jit cannot deal with numpy
        return x

    def train_transform(self, x: np.ndarray) -> torch.Tensor:
        """Apply transform to batch of optical flow image.

        Args:
            x (np.ndarray): batch of optical flow image

        Returns:
            torch.Tensor: _description_
        """
        @torch.jit.script
        def jit_trans(trans: torch.nn.Module, tt:torch.Tensor, img_dim:int):
            tt = tt.permute(0, 3, 1, 2)  # [t, h, w, c] -> [t, c, h, w]
            h, w = tt.size(2), tt.size(3)
            # InterpolationMode.BILINEAR produces slightly different results to cv2.resize(interpolation.linear).
            # see https://stackoverflow.com/questions/63519965/torch-transform-resize-vs-cv2-resize
            tt = trans(tt)
    
            tt[:, 0, :, :] = tt[:, 0, :, :] * (float(img_dim) / float(w))
            tt[:, 1, :, :] = tt[:, 1, :, :] * (float(img_dim) / float(h))
            return tt
            
        return jit_trans(self.train_trans, torch.from_numpy(x), self.img_dim)

    def val_transform(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x)


class Depth(Modality):
    """more complicated stuff.

    Args:
        Modality (): Depth modality
    """

    def __init__(self, name:str, img_crop:int, img_dim:int):
        self.name :str = name
        train_trans = torch.nn.Sequential(
            transforms.CenterCrop(size=img_crop),
            transforms.Resize([img_dim, img_dim], interpolation=InterpolationMode.BILINEAR),
        )
        self.train_trans = train_trans

    def pre_transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def train_transform(self, x: np.ndarray) -> torch.Tensor:
        """Apply transform the batch of depth image tensor of dim [t, h, w, c]. Each depth image is
        two dimensional float array.

        Args:
            x (np.ndarry): _description_

        Returns:
            torch.Tensor: transformed image tensor of size [t, c, h, w]
        """
        @torch.jit.script
        def jit_trans(trans:torch.nn.Module, tt:torch.Tensor):
            tt = trans(tt)
            tt = 1.0 / tt
            return tt

        return jit_trans(self.train_trans, torch.from_numpy(x))

    def val_transform(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x)
