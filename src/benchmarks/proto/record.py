"""Utils for decoding and encoding each item in data file
"""

import collections
import io
import json
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from .zipreader import ZipReader, is_zip_path

SIZE_UINT = 8  # bytes
ENDIAN = "little"
SUBFOLDER_NAME = "records"

CHUNK_SIZE = 1e8
# two layer processing
#   1: raw, contains everything

# todo: test speed difference between os and python level file operation


def int2bytes(num: int) -> bytes:
    """Transform a python integer to its bytes representation

    Args:
        num (int): the integer to be transformed

    Returns:
        bytes: bytes representation of num
    """
    return num.to_bytes(length=SIZE_UINT, byteorder=ENDIAN, signed=False)


class Record:
    """A protocal that stores each frame data from each sequence, i.e., data item (that consists of multilple modalities)."""

    def __init__(self, rootdir: str, features: List[str]) -> None:
        self.features : List[str] = features + ["seq_start"]  # seq_start marks if the frame is the beginning of a new sequence
        self.rootdir: str = os.path.join(rootdir, SUBFOLDER_NAME)  # store data in a separate directory

        self.byte_count: int = 0
        self.recordfile_count :int = 0
        self.recordfile_desc: int = os.open(os.path.join(self.rootdir, f"records_{self.recordfile_count}"), os.O_WRONLY)

        self.idx2recordproto: List[dict] = []
        self.idx: int = 0

    def decode_item(self, recordfile_desc: io.BufferedReader, itemproto: Dict[str, Union[int, dict]]) -> Dict[str, np.ndarray]:
        """Given a record file descript and proto of a single data item, return the
        decoded dictionary that contains feature->data of the item.

        Args:
            recordfile_desc (io.BufferedReader): python file object of the chunk file
            itemproto (Dict[str, Any]): dict that contains protocal info of a specific data item

        Returns:
            Dict[str, np.ndarray]: data
        """
        item = {}
        item_offset = itemproto["offset"]
        for feature in self.features:
            item[feature] = np.memmap(
                recordfile_desc,
                dtype=itemproto[feature]["dtype"],
                mode="r",
                offset=item_offset + itemproto[feature]["feature_offset"],
                shape=itemproto["feature"]["shape"],
            )
        # * do we need to close the memmap?
        return item

    def encode_item(self, item: Dict[str, np.ndarray], is_seq_start: bool) -> None:
        """Encode the dict(feature->np.ndarray) into bytes and store into record files.

        Args:
            item (Dict[str, np.ndarray]): feature to data (np.ndarray)
            is_start (bool): denote if the item is the beginning of a sequence
        """
        # get file name and start position for item
        self.idx2recordproto[self.idx] = {
            "recordfile_idx": self.recordfile_count,
            "item_offset": os.lseek(self.recordfile_desc, 0, os.SEEK_CUR),
            "is_seq_start": is_seq_start,
        }
        buffer = io.BytesIO()
        for feature in self.features:
            data = item[feature]
            buffer.write(data.tobytes())
            self.idx2recordproto[self.idx][feature] = {
                "dtype": data.dtype,
                "shape": data.shape,
                "feature_offset": buffer.tell(),
            }
        os.write(self.recordfile_desc, buffer.getbuffer())

        self.byte_count += buffer.tell()
        self.idx += 1

        buffer.close()
        return

    def seq_ended(self) -> None:
        """Notify the record that a complete sequence has been written, let the record
        decide if we need a new record file to write into.
        """
        if self.byte_count > CHUNK_SIZE:
            # current record file big enough
            self.byte_count = 0
            self.recordfile_count += 1
            os.close(self.recordfile_desc)
            self.recordfile_desc = os.open(os.path.join(self.rootdir, f"records_{self.recordfile_count}.bin"), flags=os.O_WRONLY)

    def save_metainfo(self, path):
        pass

    def idx4segment(self, segment_len: int, sub_features: List[str]) -> List[dict]:
        """Generate a new index mapping for dataset with: segment_len, sub_features.
        Only call this function when record has scanned all data, and record has valid attributes: rootdir, recordfile_count
        # todo unit test me!
        Args:
            segment_len (int): length of segment we are reading ! ASSUMING segment_len > 1
            sub_features: (List[str]): features (modalities data) we are reading

        Returns:
            Dict[int, Any]: idx mapping
        """

        def has_sub_features(itemproto: Dict[str, Any]) -> bool:
            """check if the current item contains all features in requested sub_features

            Args:
                item_info (Dict[str, Any]): _description_

            Returns:
                bool: _description_
            """
            # although it is numpy array, but the 'None' numpy array is a python object None
            return all(itemproto[feature] != None for feature in sub_features)

        def popleft2idx(is_segment_start:bool)-> None:
            """Popleft of the deque, add the popped item to idx4segement with flag of segment_start.

            Args:
                is_segment_start (bool): whether the item is start of segment 
            """
            temp = q.popleft()
            temp["seg_start"] = is_segment_start
            idx4segment.append(temp)
            return 

        idx4segment: List[dict] = []
        q = collections.deque()
        q_has_seg_tail = False  # indicates if the elements in queue are tail of some segment
        for i in range(self.idx):
            itemproto = self.idx2recordproto[i]
            if (not has_sub_features(itemproto)) or (itemproto["seq_start"]):
                # new seq start
                while q:
                    if q_has_seg_tail:
                        popleft2idx(is_segment_start= False)
                    else:
                        q.popleft()
                q_has_seg_tail = False
                if has_sub_features(itemproto):
                    # a valid start of sequence
                    q.append(itemproto)
            else: 
                q.append(itemproto)
                if len(q) == segment_len:
                   # claim: elements in the queue must be from the same sequence
                    popleft2idx(is_segment_start= True)
                    q_has_seg_tail = True

        # analysis (also for unit test)
        # 1. new seq (including broken) added before queue pops out
        #       the remaining elements in queue are completely useless
        # 2. new seq (including broken) added after queue has popped out
        #       the remaining elements are not start of segment but are tails of some segment
        return idx4segment

    def close(self):
        """Close opened file descriptor"""
        os.close(self.recordfile_desc)


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
        rel_path = video_frame[f"{feature}_rel_path"].strip()
        path = os.path.join(rootdir, rel_path)
        # print('root, rel_path', self.root, rel_path)
        ext = os.path.splitext(os.path.basename(path))[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif"]:
            data = zipped_pil_loader(path)
        elif ext in [".npy"]:
            data = zipped_numpy_loader(path)
        else:
            raise ValueError()
        item[feature] = data
    return item


def encode_tartanAIR(rootdir: str, json_filename: str) -> None:
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
        "frame_id",
        "image_left_rel_path",
        "depth_left_rel_path",
        "seg_left_rel_path",
        "flow_flow_rel_path",
        "flow_mask_rel_path",
    ]
    record = Record(rootdir=rootdir, features=features)
    json_path = os.path.join(rootdir, json_filename)
    with open(json_path, "r") as f:
        tartan_config = json.load(f)
    assert tartan_config["type"] == "video_pretrain"

    for video_name in tartan_config["ann"]:
        video = tartan_config["ann"][video_name]
        for i in range(len(video)):
            # only use the clips that all used data types are avaliable in each frame
            item = read_tartanair_features(rootdir, video[i], features)
            record.encode_item(item, (i == 0))
        # notify the record that a complete video sequence is read
        record.seq_ended()
    record.close()


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
