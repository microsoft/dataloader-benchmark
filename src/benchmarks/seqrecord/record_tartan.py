import io
import json
import os
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
from PIL import Image
from tqdm import tqdm

from seqrecord import SeqRecord
from zipreader import ZipReader, is_zip_path


def read_tartanair_features(rootdir: str, video_frame: Dict[str, str], features: List[str]) -> Dict[str, np.ndarray]:
    """Construct a dictionary that, for tartanAIR dataset, maps feature name to the data from corresponding modality.
    For feature names that the dict does not have values for, we store empty array.

    Args:
        rootdir (str): directory that contains data
        video_frame (Dict[str, str]): video frame that contains the relative path of each modality input
        features (List[str]): name of features to read

    Raises:
        ValueError: if the input file extension is not recognized

    Returns:
        Dict[str, np.ndarray]: map: feature name -> modality input
    """
    item = {}
    for feature in features:
        if feature not in video_frame:
            item[feature] = np.array(None)
            continue
        rel_path = video_frame[feature].strip()
        # this is only a temporary fix that only works for linux
        rel_path = "/".join(rel_path.split("/")[1:])
        path = os.path.join(rootdir, rel_path)
        # print('root, rel_path', self.root, rel_path)
        ext = os.path.splitext(os.path.basename(path))[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif"]:
            data = zipped_pil_loader(path)
            # ! added as a temporary fix, need to check if this results in array as expected
            data = np.asarray(data)
        elif ext in [".npy"]:
            data = zipped_numpy_loader(path)
        else:
            raise ValueError()
        item[feature] = data
    return item


def encode_tartanAIR(rootdir: str, json_filepath: str) -> None:
    """Given the json config, and the original data of tartanAIR, generate binary chunk file and index files.
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
        adapted from: https://github.com/yunyikristy/UPAM/blob/master/compass_mem/data/tartanair_video.py#L43
    """
    # this should include all potential attributes we intent to read
    features = [
        "image_left_rel_path",
        "depth_left_rel_path",
        "seg_left_rel_path",
        "flow_flow_rel_path",
        "flow_mask_rel_path",
    ]
    record = SeqRecord(rootdir=rootdir, features=features)
    with open(json_filepath, "r") as f:
        tartan_config = json.load(f)
    assert tartan_config["type"] == "video_pretrain"

    for video_name in tqdm(tartan_config["ann"]):
        video = tartan_config["ann"][video_name]
        # ! drop out the first frame since flow data starts at the first frame
        for i in range(1, len(video)):
            # only use the clips that all used data types are avaliable in each frame
            item = read_tartanair_features(rootdir, video[i], features)
            record.write_item(item, (i == 1))
        # notify the record that a complete video sequence is read
    record.close_recordfile()
    record.dump()


# helper functions from original tartan dataloader: https://github.com/yunyikristy/UPAM/blob/master/compass_mem/data/tartanair_video.py#L43
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


if __name__ == "__main__":
    config_path = "./tartanair.yaml"
    import yaml

    with open(config_path, mode="r") as f:
        config = yaml.safe_load(f.read())
    encode_tartanAIR(config["rootdir"], config["json_filepath"])
