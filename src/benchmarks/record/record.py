"""Utils for decoding and encoding each item in data file
"""

import collections
import io
import json
import os
import pickle
from typing import (Any, Dict, Generator, List, Optional, Sequence, Tuple,
                    TypeVar, Union)

import numpy as np
import torch.utils.data as data
from PIL import Image
from tqdm import tqdm

from zipreader import ZipReader, is_zip_path

SUBFOLDER_NAME = "records"  # subfolder name to stored serialized record data
MAX_RECORDFILE_SIZE = 1e8  # 100 mb, maximum size of a single record file

# todo: test speed difference between os and python level file operation
# todo :frame 0, optical flow offset
# todo: pre transform

R = TypeVar("R", bound="Record")


class Record:
    """A serialization protocal that stores each frame data from each sequence as raw bytes, while provides index files to
    read the serialized dataset."""

    def __init__(self, rootdir: str, features: List[str]) -> None:
        self.features: List[str] = features  # seq_start marks if the frame is the beginning of a new sequence
        self.rootdir: str = os.path.join(rootdir, SUBFOLDER_NAME)  # store data in a separate directory (subfolder)
        os.makedirs(self.rootdir, exist_ok=True)

        self.byte_count: int = 0  # number of bytes written into current record file
        self.recordfile_idx: int = 0  # number of record file created for dataset
        self.recordfile_endpoints: List[list] = [
            [0]
        ]  # track the idx endpoints for each record file, [[start_idx, end_idx)]
        self.recordfile_desc: Optional[int] = None  # file descriptor for current record file
        self.idx2recordproto: Dict[int, dict] = {}  # serialization proto info of each data item
        self.idx: int = 0  # index of current data item to be processed

        # a cache dict that stores protocal info for each (segment_len, sub features)
        self.segmentproto_cache: Dict[str, dict] = {}

    def recordfile_idx_to_path(self, recordfile_idx: int) -> str:
        """Return path to record file given the idx of the record file.

        Args:
            recordfile_idx (int): idx of the record file

        Returns:
            str: path to record file
        """
        return os.path.join(self.rootdir, f"records_{recordfile_idx}.bin")

    def get_recordfiles(self) -> List[str]:
        """Return a list of paths of record files

        Returns:
            List[str]: list of paths to record files
        """
        return [self.recordfile_idx_to_path(i) for i in range(self.recordfile_idx)]

    def encode_item(self, item: Dict[str, np.ndarray], is_seq_start: bool) -> None:
        """Encode the data dict(feature->np.ndarray) into bytes and write encoded bytes into current record files.

        Args:
            item (Dict[str, np.ndarray]): feature to data (np.ndarray)
            is_seq_start (bool): denote if the item is the beginning of a sequence
        """
        if is_seq_start:
            self.seq_start()
        # get file name and start position for item
        self.idx2recordproto[self.idx] = {
            "item_idx": self.idx,
            "recordfile_idx": self.recordfile_idx,
            "item_offset": os.lseek(self.recordfile_desc, 0, os.SEEK_CUR),
            "is_seq_start": is_seq_start,
        }
        buffer = io.BytesIO()
        for feature in self.features:
            data = item[feature]
            buffer.write(data.tobytes())
            self.idx2recordproto[self.idx][feature] = {
                "is_none": (data.dtype == np.dtype("O") and data == None),  # this feature is essentially missing, and
                "dtype": data.dtype,
                "shape": data.shape,
                "feature_offset": buffer.tell(),
            }
        os.write(self.recordfile_desc, buffer.getbuffer())

        self.byte_count += buffer.tell()
        self.idx += 1

        buffer.close()
        return

    def seq_start(self) -> None:
        """Notify the record that a new sequence is being written, let the record
        decide if we need a new record file to write into.
        Two cases we need to open new file:
            1. we currently do not have record file to write into
            2. current file size is big enough (larger than MAX_RECORDFILE_SIZE)
        """
        if self.byte_count > MAX_RECORDFILE_SIZE:
            # current record file big enough
            self.byte_count = 0
            self.recordfile_idx += 1
            self.recordfile_endpoints[-1].append(self.idx)
            self.recordfile_endpoints.append([self.idx])
            os.close(self.recordfile_desc)
            self.recordfile_desc = os.open(
                self.recordfile_idx_to_path(self.recordfile_idx), flags=os.O_WRONLY | os.O_CREAT
            )
        elif self.recordfile_desc == None:
            # no opened record file to write into
            self.recordfile_desc = os.open(
                self.recordfile_idx_to_path(self.recordfile_idx), flags=os.O_WRONLY | os.O_CREAT
            )

    def decode_item(
        self, recordfile_desc: io.BufferedReader, itemproto: Dict[str, Union[int, dict]]
    ) -> Dict[str, np.ndarray]:
        """Given record file descriptor and serialization proto of a single data item, return the
        decoded dictionary(feature->data(np.ndarray)) of the item.

        Args:
            recordfile_desc (io.BufferedReader): python file object of the record file (required by numpy)
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

    def decode_record(self) -> Generator[Dict[str, np.ndarray], None, None]:
        """Given that the dataset has been recored, decode the record sequentially, each time returning a dict that contains the data
        item.

        Yields:
            Generator[Dict[str, np.ndarray], None, None]: data item [feature->data]. All data items are being returned sequentially
        """
        for i in range(self.recordfile_idx):
            recordfile_path = self.recordfile_idx_to_path(i)
            endpoints = self.recordfile_endpoints[i]
            with open(recordfile_path, mode="rb") as f:
                for idx in range(endpoints[0], endpoints[1]):
                    yield self.decode_item(f, self.idx2recordproto[idx])

    def get_proto4segment(self, segment_len: int, sub_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate a new index2recordproto so as to read segments from sequences of datasets. Each data item of segment should
        contain all features in sub_features.

        Note:
        Only call this function when record has scanned all data in dataset, and record has valid attributes: rootdir, recordfile_idx

        Args:
            segment_len (int): length of segment we are reading, 1< segment_len < sequence length
            sub_features: (Optional[List[str]]): features (modalities data) we need for each data item in segment to contain. If it is None,
            then we read all features.

        Returns:
            Dict[str, Any]: protocal needed for reading segments from data 
        """

        def has_sub_features(itemproto: Dict[str, Any]) -> bool:
            """check if the current item contains all features requested in sub_features

            Args:
                itemproto (Dict[str, Any]): item proto to be excamined

            Returns:
                bool:
            """
            return all(not itemproto[feature]["is_none"] for feature in sub_features)

        def popleft2idx(is_segment_start: bool) -> None:
            """Popleft of the deque, add the popped item to idx4segement with flag of is_segment_start.

            Args:
                is_segment_start (bool): whether the item is start of segment
            """
            temp = q.popleft()
            temp["is_seg_start"] = is_segment_start
            if is_segment_start:
                headidx4segment.append(len(itemidx4segment))
            itemidx4segment.append(temp)
            return

        sub_features = self.features if sub_features == None else sub_features
        cache_key = str(segment_len) + "#" + "#".join(sorted(sub_features))
        if cache_key in self.segmentproto_cache:
            return self.segmentproto_cache[cache_key]
        itemidx4segment: List[dict] = []
        headidx4segment: List[int] = []
        q = collections.deque()
        q_has_seg_tail = False  # indicates if the elements in queue are tail of some segment
        for i in range(self.idx):
            itemproto = self.idx2recordproto[i]
            if (not has_sub_features(itemproto)) or (itemproto["is_seq_start"]):
                # new seq start
                while q:
                    if q_has_seg_tail:
                        popleft2idx(is_segment_start=False)
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
                    popleft2idx(is_segment_start=True)
                    q_has_seg_tail = True

        if q and q_has_seg_tail:
            # front element in queue is need as last element of some segment
            popleft2idx(is_segment_start=False)

        # 1. new seq (including broken) added before queue pops out
        #       the remaining elements in queue are completely useless
        # 2. new seq (including broken) added after queue has popped out
        #       the remaining elements are not start of segment but are tails of some segment
        self.segmentproto_cache[cache_key] = {"segment_len": segment_len, "features": sub_features, "head_idx_proto": headidx4segment, "item_idx_proto":itemidx4segment} 
        return self.segmentproto_cache[cache_key] 

    def decode_segment(self, start_item_idx:int, end_item_idx:int, segment_len:int) -> Generator[Dictp[str, np.ndarray], None, None]:
        """Given the range of items, return a sequence of segments in between with length segment_len.
        Note:
            The start_item_idx, and end_item_idx must be compatible(generated from) with the segment length.

        Args:
            start_item_idx (int): item_idx of the head of the first segment to return, inclusive
            end_item_idx (int): item_idx of the head of the last segment to return, exclusive.
            segment_len (int): length of segment we need to generate

        Yields:
            Generator[Dict[str, np.ndarray], None, None]: sequence of items of length segment_len 
e       """
        recordfile_start, recordfile_end = -1, -1
        for i in range(self.recordfile_idx):
            endpoints = self.recordfile_endpoints[i]
            # both endpoints[1] and end_item_idx is exclusive
            if recordfile_start == -1 and (endpoints[0] <= start_item_idx < endpoints[1]):
                recordfile_start = i
            if recordfile_end == -1 and (endpoints[0] < end_item_idx <= endpoints[1]):
                recordfile_end= i
        for i in range(recordfile_start, recordfile_end+1):
            recordfile_path = self.recordfile_idx_to_path(i)
            endpoints = self.recordfile_endpoints[i]
            endpoints[0] = max(endpoints[0], start_item_idx)
            endpoints[1] = min(endpoints[1], end_item_idx)
            q = collections.deque()
            with open(recordfile_path, mode="rb") as f:
                for idx in range(endpoints[0], endpoints[1]):
                    q.append(self.decode_item(f, self.idx2recordproto[idx]))
                    while not q[0]["is_seg_start"]:
                        q.popleft()
                    if len(q) == segment_len:
                        yield self.collate_items(q)

    def collate_items(self, q:Sequence[dict]) -> Dict[str, np.ndarray]:
        segment = {}
        for feature in self.features:
            segment[feature] = np.stack([item[feature] for item in q], axis=0)
        return segment

    def close_recordfile(self):
        """Close opened file descriptor! This needs to be called when finishes scanning over the dataset."""
        self.recordfile_endpoints[-1].append(self.idx)
        os.close(self.recordfile_desc)

    def dump(self) -> None:
        """save instance of record"""
        with open(os.path.join(self.rootdir, "recordproto.bin"), mode="wb") as f:
            pickle.dump(self, file=f)

    @classmethod
    def load_recordproto(cls, path: str) -> R:
        """return an instance of record from file (stored at path).

        Args:
            path (str): path to the file that stores pickled record

        Returns:
            R: an instance of record
        """
        with open(path, mode="wb") as f:
            return pickle.load(f)
    
    def shard_record(self, num_shards:int):



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
    record = Record(rootdir=rootdir, features=features)
    with open(json_filepath, "r") as f:
        tartan_config = json.load(f)
    assert tartan_config["type"] == "video_pretrain"

    for video_name in tqdm(tartan_config["ann"]):
        video = tartan_config["ann"][video_name]
        for i in range(len(video)):
            # only use the clips that all used data types are avaliable in each frame
            item = read_tartanair_features(rootdir, video[i], features)
            record.encode_item(item, (i == 0))
        # notify the record that a complete video sequence is read
    record.close_recordfile()


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
