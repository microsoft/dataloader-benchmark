"""A TartanAIR dataset (both iterative and map style provide)
"""


from time import perf_counter
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
# todo: test with old dataloader and some stats of record (number of record comparison etc...)


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

    def __getitem__(self, index: int) -> Dict[str, List[np.ndarray]]:
        head_idx = self.segmentproto["head4segment"][index]
        item = self.record.read_one_segment(self.segmentproto["segment_len"], head_idx)
        return item

    def __len__(self):
        return len(self.segmentproto["head4segment"])


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    collated_batch = {}
    for feature in batch[0]:
        collated_batch[feature] = torch.stack([batch[i][feature] for i in range(len(batch))], axis=0)
    return collated_batch


def list2array(data_np: Dict[str, List[np.ndarray]]) -> Dict[str, torch.Tensor]:
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


def build_itertartanair_datapipe(record, segment_len, dl_config):

    datapipe = IterTartanAIRDatapipe(record, segment_len, features=None)
    # Shuffle will happen as long as you do NOT set `shuffle=False` later in the DataLoader
    # https://pytorch.org/data/main/tutorial.html#working-with-dataloader
    datapipe = dp.iter.Shuffler(datapipe, buffer_size=100)
    # sharding: Place ShardingFilter (datapipe.sharding_filter) as early as possible in the pipeline,
    # especially before expensive operations such as decoding, in order to avoid repeating these expensive operations across worker/distributed processes.
    datapipe = dp.iter.ShardingFilter(datapipe)
    datapipe = dp.iter.Mapper(datapipe, fn=list2array)
    # Note that if you choose to use Batcher while setting batch_size > 1 for DataLoader,
    # your samples will be batched more than once. You should choose one or the other.
    # https://pytorch.org/data/main/tutorial.html#working-with-dataloader
    datapipe = dp.iter.Batcher(datapipe, batch_size=dl_config["batch_size"], drop_last=True)
    datapipe = dp.iter.Collator(datapipe)
    return datapipe


def build_maptartanair_datapipe(record, segment_len, batch_size):
    # ! sharding /distributed? for mapping style datapipe?

    datapipe = MapTartanAIRDatapipe(record, segment_len, features=None)
    datapipe = dp.map.Shuffler(datapipe)
    datapipe = dp.map.Mapper(datapipe, fn=list2array)
    datapipe = dp.map.Batcher(datapipe, batch_size=batch_size, drop_last=True)
    datapipe = dp.map.Mapper(datapipe, fn=collate_fn)
    return datapipe


def test_iter(record, segment_len, dl_config):
    datapipe = build_itertartanair_datapipe(record, segment_len, dl_config)
    dataloader = torch.utils.data.DataLoader(
        datapipe, shuffle=True, num_workers=dl_config["num_workers"], prefetch_factor=dl_config["prefetch_factor"]
    )
    for batch in tqdm(dataloader):
        # is this the best way? we should do one-pass to 'small network'
        for key in batch:
            batch[key].cuda()


def test_map(record, segment_len, dl_config):
    datapipe = build_maptartanair_datapipe(record, segment_len, dl_config["batch_size"])
    dataloader = torch.utils.data.DataLoader(
        datapipe, shuffle=True, num_workers=dl_config["num_workers"], prefetch_factor=dl_config["prefetch_factor"]
    )
    for batch in tqdm(dataloader):
        # # for key in batch:
        # for key in batch:
        #     batch[key].cuda()
        pass


if __name__ == "__main__":
    # sanity check dataloader
    rootdir = "/datadrive/azure_mounted_data/commondataset/tartanair-release1/abandonedfactory/records"
    record = SeqRecord.load_recordobj(rootdir)
    segment_len = 16

    dl_config = {"num_workers": 0, "batch_size": 8, "prefetch_factor": 2}
    # start_iter = perf_counter()
    # test_iter(record, segment_len, dl_config)
    # end_iter = perf_counter()
    # print(f"{end_iter - start_iter =}")
    start_map = perf_counter()
    test_map(record, segment_len, dl_config)
    end_map = perf_counter()
    print(f"{end_map - start_map =}")
