"""A TartanAIR dataset (both iterative and map style provide)
"""


from typing import Dict, List, Optional

import numpy as np
import torch
import torchdata.datapipes as dp
from tqdm import tqdm

from seqrecord import SeqRecord

# todo: pre fetch and shuffle
# todo: transform (offline, online, to discuss with shuang)
# todo: add profile
# todo: ask, how distributed works?


class IterTartanAIRDatapipe(dp.iter.IterDataPipe):
    """A torch datapiple class that iteratively read from record files."""

    def __init__(self, record: SeqRecord, segment_len: int, features: Optional[List[str]]) -> None:
        super().__init__()
        self.segmentproto = record.get_proto4segment(segment_len, features)
        self.record = record

    def __iter__(self):
        for segment in self.record.read_all_segments(self.segmentproto):
            yield segment

    def __len__(self):
        return len(self.segmentproto["head4segment"])


class MapTartanAIRDatapipe(dp.map.MapDataPipe):
    def __init__(self, record: SeqRecord, segment_len: int, features: Optional[List[str]]) -> None:
        super().__init__()
        self.segmentproto = record.get_proto4segment(segment_len, features)
        self.record = record

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        head_idx = self.segmentproto["head4segment"][index]
        data_np_arrays = self.record.read_one_segment(self.segmentproto["segment_len"], head_idx)
        return stack_nparray(data_np_arrays)

    def __len__(self):
        return len(self.segmentproto["head4segment"])


def stack_nparray(data_np: Dict[str, List[np.ndarray]]) -> Dict[str, torch.Tensor]:
    """transform data from list of np.array to torch tensor

    Args:
        data_np (Dict[str, List[np.ndarray]]): _description_

    Returns:
        Dict[str, torch.Tensor]: _description_
    """
    data_torch: Dict[str, torch.Tensor] = {}
    for feature in data_np:
        data_torch[feature] = torch.from_numpy(np.stack(data_np[feature], axis=0))
    return data_torch


def build_itertartanair_datapipe(record, segment_len):

    datapipe = IterTartanAIRDatapipe(record, segment_len, features=None)
    # Shuffle will happen as long as you do NOT set `shuffle=False` later in the DataLoader
    # https://pytorch.org/data/main/tutorial.html#working-with-dataloader
    datapipe = dp.iter.Shuffler(datapipe, buffer_size=100)
    # sharding: Place ShardingFilter (datapipe.sharding_filter) as early as possible in the pipeline,
    # especially before expensive operations such as decoding, in order to avoid repeating these expensive operations across worker/distributed processes.
    datapipe = dp.iter.ShardingFilter(datapipe)
    datapipe = dp.iter.Mapper(datapipe, fn=stack_nparray)
    # Note that if you choose to use Batcher while setting batch_size > 1 for DataLoader,
    # your samples will be batched more than once. You should choose one or the other.
    # https://pytorch.org/data/main/tutorial.html#working-with-dataloader
    datapipe = dp.iter.Batcher(datapipe, batch_size=8, drop_last=True)
    datapipe = dp.iter.Collator(datapipe)
    return datapipe


def build_maptartanair_datapipe(record, segment_len):
    # ! sharding /distributed? for mapping style datapipe?

    datapipe = MapTartanAIRDatapipe(record, segment_len, features=None)
    datapipe = dp.map.Shuffler(datapipe)
    datapipe = dp.map.Mapper(datapipe, fn=stack_nparray)
    datapipe = dp.map.Batcher(datapipe, batch_size=8, drop_last=True)
    return datapipe


if __name__ == "__main__":
    # sanity check dataloader
    rootdir = "/datadrive/azure_mounted_data/commondataset/tartanair-release1/abandonedfactory/records"
    record = SeqRecord.load_recordobj(rootdir)
    segment_len = 16
    datapipe = build_itertartanair_datapipe(record, segment_len)
    dataloader = torch.utils.data.DataLoader(datapipe, shuffle=True, num_workers=0, prefetch_factor=2)
    num_segs = 0
    for segment in tqdm(dataloader):
        num_segs += 1
    print(num_segs)
