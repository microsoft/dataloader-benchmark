import io
import json
import os
from collections import defaultdict

import numpy as np
import torch.utils.data as data
from PIL import Image

from .zipreader import ZipReader, is_zip_path


class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = f"Maks: total patches {self.total_patches}, mask patches {self.total_masks}"
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack(
            [
                np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
                np.ones(self.num_masks_per_frame),
            ]
        )
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames, 1)).flatten()
        return mask


def zipped_pil_loader(path):
    """PIL image loader that supports zipped images.

    Ref: https://github.com/SwinTransformer/Transformer-SSL/blob/main/data/cached_image_folder.py#L179
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    if isinstance(path, bytes):
        img = Image.open(io.BytesIO(path))
    elif is_zip_path(path):
        data = ZipReader.read(path)
        img = Image.open(io.BytesIO(data))
    else:
        with open(path, "rb") as f:
            img = Image.open(f)
    return img.convert("RGB")


def zipped_numpy_loader(path):
    """NumPy loader that supports zipped files."""
    if isinstance(path, bytes):
        x = np.load(io.BytesIO(path))
    elif is_zip_path(path):
        data = ZipReader.read(path)
        x = np.load(io.BytesIO(data))
    else:
        with open(path, "rb") as f:
            x = np.load(f)
    return x


class TartanAirVideoDataset(data.Dataset):
    """A TartanAir video dataset where the annotations are arranged in this way: ::
        Format:
        {
            'type': 'video',
            'ann': {
                'video name 1': [
                    {'image_left_rel_path': image_left_rel_path,
                     'depth_left_rel_path': depth_left_rel_path,
                     'seg_left_rel_path': seg_left_rel_path,
                     'flow_flow_rel_path': flow_flow_rel_path,
                     'flow_mask_rel_path': flow_mask_rel_path},
                    ...
                ]
                ...
            }
        }
    Args:
        root (string): Root directory path.
        ann_file (string): Annotation file name.
        clip_len (int): Number of frames in each sampled clip.
        modalities (List[str]): Data types to include in the sample.
        transform (callable): A function/transform that takes in
            a sampled clip and returns a transformed version.
     Attributes:
        clip_indices (list): List of (video_name, start_frame_index) tuples
    """

    # @profile
    def __init__(
        self,
        ann_file="",
        clip_len=8,
        seq_len=3,
        modalities=["image_left", "flow_flow"],
        transform=None,
        video_name_keyword=None,
        ffcv=False,
        return_mask_position=False,
    ):
        # Load annotation file.
        root = os.path.dirname(os.path.abspath(ann_file))
        ann_path = os.path.join(ann_file)
        with open(ann_path) as f:
            self.ann = json.load(f)
        assert self.ann["type"] == "video_pretrain"

        # Generate clip indices. Format: (video name, start frame index).
        self.clip_indices = []
        for video_name in self.ann["ann"]:
            if video_name_keyword:
                if video_name_keyword not in video_name:
                    continue
            video = self.ann["ann"][video_name]
            if len(video) >= clip_len * seq_len:
                for start_frame_index in range(len(video) - clip_len * seq_len + 1):
                    # Only use the clips that all used data types are available in each frame.
                    end_frame_index = start_frame_index + clip_len * seq_len - 1
                    if all(
                        f"{modality}_rel_path" in video[i]
                        for modality in modalities
                        for i in range(start_frame_index, end_frame_index + 1)
                    ):
                        self.clip_indices.append((video_name, start_frame_index))

        # Other settings.
        self.root = root
        self.seq_len = seq_len
        self.modalities = modalities
        self.transform = transform
        self.num_seq = clip_len
        self.return_mask_position = return_mask_position
        self.ffcv = ffcv

        # added for mae pretrain
        if self.return_mask_position:
            self.masked_position_generator = TubeMaskingGenerator((16 // 2, 224 // 16, 224 // 16), 0.9)

    # @profile
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            item: An item dictionary containing all the annotations for the clip
                  Format: {'image_left': image_left, 'flow_flow': flow_flow, ...}
        """
        # Get annotation.
        video_name, start_frame_index = self.clip_indices[index]

        item = defaultdict(list)
        for frame_index in range(start_frame_index, start_frame_index + self.num_seq * self.seq_len):
            frame_ann = self.ann["ann"][video_name][frame_index]

            # Load data.
            for modality in self.modalities:
                rel_path = frame_ann[f"{modality}_rel_path"].strip()
                path = os.path.join(self.root, rel_path)
                # print('root, rel_path', self.root, rel_path)
                ext = os.path.splitext(os.path.basename(path))[1].lower()
                if ext in [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif"]:
                    data = zipped_pil_loader(path)
                elif ext in [".npy"]:
                    data = zipped_numpy_loader(path)
                else:
                    raise ValueError()

                item[modality].append(data)

        if self.ffcv:
            for key in item:
                item[key] = np.asarray(item[key])

            # todo ffcv + mask  is currently not tested
            if self.masked_position_generator:
                mask = self.masked_position_generator()
                return (*item.values(), mask)

            else:
                return (*item.values(),)

        else:
            assert self.transform is not None

            item = self.transform(
                item
            )  # Note: The transform function should transform data and stack them properly for each data type.

            #### added for COMPASS as  N, C, SL, H, W ########
            for modality, _ in item.items():
                seq = item[modality]
                _, C, H, W = seq.permute(1, 0, 2, 3).shape
                # (C, H, W) = seq[0].size()
                # seq = torch.stack(seq, 0)
                seq = seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1, 2)
                item[modality] = seq  # N, C, SL, H,  W

            if self.return_mask_position:
                mask = self.masked_position_generator()
                return item, mask
            else:
                return item

    def __len__(self):
        return len(self.clip_indices)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += f"    Number of datapoints: {self.__len__()}\n"
        fmt_str += f"    Root Location: {self.root}\n"
        return fmt_str
