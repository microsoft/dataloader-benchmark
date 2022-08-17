"""Utils for decoding and encoding each item in data file
"""

import collections
import io
import os
import pickle
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np

SUBFOLDER_NAME = "records"  # subfolder name to stored serialized record data
MAX_RECORDFILE_SIZE = 1e8  # 1e8, 100 mb, maximum size of a single record file

# todo: pre transform
# todo: test read segment with multiple workers: number of overall segments and start_end of each segment
# and other configs along with bottle necker of torch and other tools
# skip the first frame in record
# todo: try file blob
# todo: test speed difference between os and python level file operation
# todo: add buffer for reading

SR = TypeVar("R", bound="SeqRecord")


class SeqRecord:
    """A serialization protocal that stores sequences into record files, while provides index files to
    read segments from records."""

    def __init__(self, rootdir: str, features: List[str]) -> None:
        self.features: List[str] = features  # seq_start marks if the frame is the beginning of a new sequence
        self.rootdir: str = os.path.join(rootdir, SUBFOLDER_NAME)  # store data in a separate directory (subfolder)
        os.makedirs(self.rootdir, exist_ok=True)

        self.byte_count: int = 0  # number of bytes written into current record file
        self.recordfile_idx: int = 0  # number of record file created for dataset
        # track the idx endpoints for each record file, [[start_idx, end_idx]], both are inclusive
        self.recordfile_endpoints: List[list] = []
        self.recordfile_desc: Optional[int] = None  # file descriptor for current record file
        self.idx2recordproto: Dict[int, dict] = {}  # serialization proto info of each data item
        self.idx: int = 0  # index of current data item to be processed

        # a cache dict that stores protocal info for each (segment_len, sub features)
        self.segmentproto_cache: Dict[str, dict] = {}

    def recordfile_idx_to_path(self, recordfile_idx: int) -> str:
        return os.path.join(self.rootdir, f"records_{recordfile_idx}.bin")

    def get_recordfiles(self) -> List[str]:
        return [self.recordfile_idx_to_path(i) for i in range(self.recordfile_idx)]

    def write_item(self, item: Dict[str, np.ndarray], is_seq_start: bool) -> None:
        """write one item data dict(feature->np.ndarray) into bytes and write encoded bytes into current record files.

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
            self.idx2recordproto[self.idx][feature] = {
                "is_none": (data.dtype == np.dtype("O") and data == None),  # this feature is essentially missing, and
                "dtype": data.dtype,
                "shape": data.shape,
                "feature_offset": buffer.tell(),
                "nbytes": data.nbytes,
            }
            buffer.write(data.tobytes())
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
            self.recordfile_endpoints[-1].append(self.idx - 1)
            self.recordfile_endpoints.append([self.idx])
            os.close(self.recordfile_desc)
            self.recordfile_desc = os.open(
                self.recordfile_idx_to_path(self.recordfile_idx), flags=os.O_WRONLY | os.O_CREAT
            )
        elif self.recordfile_desc == None:
            # no opened record file to write into
            self.recordfile_endpoints.append([self.idx])
            self.recordfile_desc = os.open(
                self.recordfile_idx_to_path(self.recordfile_idx), flags=os.O_WRONLY | os.O_CREAT
            )

    def read_item(
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
        item_offset = itemproto["item_offset"]
        for feature in self.features:
            item[feature] = np.memmap(
                recordfile_desc,
                dtype=itemproto[feature]["dtype"],
                mode="r",
                offset=item_offset + itemproto[feature]["feature_offset"],
                shape=itemproto[feature]["shape"],
            )
        # * do we need to close the memmap?
        return item

    def read_item_frombuffer(
        self, recordfile_desc: io.BufferedReader, itemproto: Dict[str, Union[int, dict]]
    ) -> Dict[str, np.ndarray]:
        """Given record file descriptor and serialization proto of a single data item, return the
        decoded dictionary(feature->data(np.ndarray)) of the item, where decoding is done by np.frombuffer()

        Args:
            recordfile_desc (io.BufferedReader): python file object of the record file (required by numpy)
            itemproto (Dict[str, Any]): dict that contains protocal info of a specific data item

        Returns:
            Dict[str, np.ndarray]: data
        """
        item = {}
        recordfile_desc.seek(itemproto["item_offset"])
        for feature in self.features:
            bytes = recordfile_desc.read(itemproto[feature]["nbytes"])
            array1d = np.frombuffer(
                bytes,
                dtype=itemproto[feature]["dtype"],
            )
            item[feature] = array1d.reshape(itemproto[feature]["shape"])
        return item

    def read_record(self) -> Generator[Dict[str, np.ndarray], None, None]:
        """Given that the dataset has been recored, decode the record sequentially, each time returning a dict that contains the data
        item.

        Yields:
            Generator[Dict[str, np.ndarray], None, None]: data item [feature->data]. All data items are being returned sequentially
        """
        for i in range(self.recordfile_idx):
            recordfile_path = self.recordfile_idx_to_path(i)
            endpoints = self.recordfile_endpoints[i]
            with open(recordfile_path, mode="rb") as f:
                for idx in range(endpoints[0], endpoints[1] + 1):
                    yield self.read_item(f, self.idx2recordproto[idx])

    def get_proto4segment(self, segment_len: int, sub_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate a protocal for reading segments from records. Each data item of segment should
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
            return all(not itemproto[feature]["is_none"] for feature in sub_features)

        def update_segmentproto(item_idx: int, is_segment_start: bool) -> None:
            if is_segment_start:
                head4segment.append(item_idx)
            recordfile_idx = self.idx2recordproto[item_idx]["recordfile_idx"]
            file2segment_items[recordfile_idx].append((is_segment_start, item_idx))
            return

        sub_features = self.features if sub_features == None else sub_features
        cache_key = str(segment_len) + "#" + "#".join(sorted(sub_features))
        if cache_key in self.segmentproto_cache:
            return self.segmentproto_cache[cache_key]
        head4segment: List[int] = []
        file2segment_items: dict[int, List[Tuple[bool, int]]] = collections.defaultdict(list)
        q = collections.deque()
        q_has_seg_tail = False  # indicates if the elements currently in queue are tail of some segment
        for idx in range(self.idx):
            itemproto = self.idx2recordproto[idx]
            if (not has_sub_features(itemproto)) or (itemproto["is_seq_start"]):
                # new seq start
                while q:
                    if q_has_seg_tail:
                        update_segmentproto(q.popleft(), is_segment_start=False)
                    else:
                        q.popleft()
                q_has_seg_tail = False
                if has_sub_features(itemproto):
                    # a valid start of sequence
                    q.append(idx)
            else:
                q.append(idx)
                if len(q) == segment_len:
                    # claim: elements in the queue must be from the same sequence
                    update_segmentproto(q.popleft(), is_segment_start=True)
                    q_has_seg_tail = True

        if q and q_has_seg_tail:
            # front element in queue is need as last element of some segment
            update_segmentproto(q.popleft(), is_segment_start=False)

        # 1. new seq (including broken) added before queue pops out
        #       the remaining elements in queue are completely useless
        # 2. new seq (including broken) added after queue has popped out
        #       the remaining elements are not start of segment but are tails of some segment
        self.segmentproto_cache[cache_key] = {
            "segment_len": segment_len,
            "features": sub_features,
            "head4segment": head4segment,
            "file2segment_items": file2segment_items,
        }
        return self.segmentproto_cache[cache_key]

    def read_all_segments(self, segment_proto: dict):
        """Iterate through the whole records and return segments sequential.

        Yields:
            _type_: _description_
        """
        segment_len = segment_proto["segment_len"]
        for recordfile_idx, item_list in segment_proto["file2segment_items"].items():
            recordfile_path = self.recordfile_idx_to_path(recordfile_idx)
            q = collections.deque()
            with open(recordfile_path, mode="rb") as f:
                for is_segment_start, item_idx in item_list:
                    q.append((is_segment_start, self.read_item(f, self.idx2recordproto[item_idx])))
                    while not q[0][0]:
                        q.popleft()
                    if len(q) == segment_len:
                        yield self.collate_items(q)
                        q.popleft()

    def read_one_segment(
        self,
        segment_len: int,
        head_idx: int,
    ) -> Dict[str, List[np.ndarray]]:
        """Read a segment (of lenght segment_len) starting from the item index being head_idx.

        Args:
            segment_len (int): length of segment we need to generate
            head_idx (int): item_idx of the head of the segment to be read.

        Returns:
            Dict[str, np.ndarray]: segment data
        """
        recordfile_path = self.recordfile_idx_to_path(self.idx2recordproto[head_idx]["recordfile_idx"])
        q = []
        with open(recordfile_path, mode="rb") as f:
            for idx in range(head_idx, head_idx + segment_len):
                q.append((idx == head_idx, self.read_item_frombuffer(f, self.idx2recordproto[idx])))
        return self.collate_items(q)

    def collate_items(self, q: Sequence[Tuple[bool, dict]]) -> Dict[str, List[np.ndarray]]:
        segment = {}
        for feature in self.features:
            segment[feature] = [item[feature] for _, item in q]
        return segment

    def close_recordfile(self):
        """Close opened file descriptor! This needs to be called when finishes scanning over the dataset."""
        self.recordfile_endpoints[-1].append(self.idx - 1)
        self.recordfile_idx += 1
        os.close(self.recordfile_desc)

    def dump(self) -> None:
        """save instance of record"""
        with open(os.path.join(self.rootdir, "recordproto.bin"), mode="wb") as f:
            pickle.dump(self, file=f)

    @classmethod
    def load_recordobj(cls, rootdir: str) -> SR:
        """return an instance of sequence record from file (stored at path).

        Args:
            path (str): path to the file that stores pickled record

        Returns:
            R: an instance of record
        """
        with open(os.path.join(rootdir, "recordproto.bin"), mode="rb") as f:
            obj = pickle.load(f)
        return obj
