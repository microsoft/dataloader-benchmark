"""A TartanAIR dataset (both iterative and map style provide)
"""


from time import perf_counter
from typing import Dict, List, Optional

import numpy as np
import torch
import torchdata.datapipes as dp
from src.benchmarks.seqrecord.dataload_utils import InputConfig
from src.benchmarks.seqrecord.seqrecord import SeqRecord
from tqdm import tqdm

# todo: how to shard in the record file level? future work and why it is needed?
# just break read_all_segments and return record with subset of record files, along with file2item index


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

    def __getitm__(self, idx: int) -> Dict[str, np.ndarray]:
        head_idx = self.segmentproto["head4segment"][idx]
        item = self.record.read_one_segment(self.segmentproto["segment_len"], head_idx)
        return list2array(item)

    def __len__(self):
        return len(self.segmentproto["head4segment"])


def collate_fn(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    collated_batch: Dict[str, torch.Tensor] = {}
    for feature in batch[0]:
        collated_batch[feature] = torch.from_numpy(np.stack([batch[i][feature] for i in range(len(batch))], axis=0))
    return collated_batch


def list2array(data_list: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    """transform data from list of np.array to a single numpy array

    Args:
        data_np (Dict[str, List[np.ndarray]]): _description_

    Returns:
        Dict[str, np.ndarray]: _description_
    """
    data_array: Dict[str, np.ndarray] = {}
    for feature in data_list:
        data_array[feature] = np.stack(data_list[feature], axis=0)
    return data_array


def build_itertartanair_datapipe(record, segment_len, dl_config, input_config):

    datapipe = IterTartanAIRDatapipe(record, segment_len, None)
    # Shuffle will happen as long as you do NOT set `shuffle=False` later in the DataLoader
    # https://pytorch.org/data/main/tutorial.html#working-with-dataloader
    datapipe = dp.iter.Shuffler(datapipe, buffer_size=100)
    # sharding: Place ShardingFilter (datapipe.sharding_filter) as early as possible in the pipeline,
    # especially before expensive operations such as decoding, in order to avoid repeating these expensive operations across worker/distributed processes.
    datapipe = dp.iter.ShardingFilter(datapipe)
    datapipe = dp.iter.Mapper(datapipe, fn=list2array)
    datapipe = dp.iter.Mapper(datapipe, fn=input_config.train_transform)
    # Note that if you choose to use Batcher while setting batch_size > 1 for DataLoader,
    # your samples will be batched more than once. You should choose one or the other.
    # https://pytorch.org/data/main/tutorial.html#working-with-dataloader
    datapipe = dp.iter.Batcher(datapipe, batch_size=dl_config["batch_size"], drop_last=True)
    datapipe = dp.iter.Collator(datapipe, collate_fn=collate_fn)
    return datapipe


def build_maptartanair_datapipe(record, segment_len, dl_config):
    datapipe = MapTartanAIRDatapipe(record, segment_len, None)
    datapipe = dp.map.Shuffler(datapipe)
    datapipe = dp.map.Batcher(datapipe, batch_size=dl_config["batch_size"])
    datapipe = dp.map.Mapper(datapipe, fn=collate_fn)
    return datapipe


def test_iter(record, segment_len, dl_config, input_config):
    datapipe = build_itertartanair_datapipe(record, segment_len, dl_config, input_config)
    dataloader = torch.utils.data.DataLoader(
        datapipe, shuffle=True, num_workers=dl_config["num_workers"], prefetch_factor=dl_config["prefetch_factor"]
    )
    for batch in tqdm(dataloader):
        # is this the best way? we should do one-pass to 'small network'
        for key in batch:
            batch[key].cuda()
        torch.cuda.synchronize()


def test_map(record, segment_len, dl_config):
    dataset = MapTartanAIRDatapipe(record, segment_len, features=None)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=dl_config["batch_size"],
        shuffle=True,
        num_workers=dl_config["num_workers"],
        drop_last=True,
    )
    for batch in tqdm(dataloader):
        for key in batch:
            batch[key].cuda()
        torch.cuda.synchronize()


if __name__ == "__main__":

    rootdir = "/datadrive/azure_mounted_data/commondataset2/tartanair-release1/abandonedfactory/records"
    record = SeqRecord.load_record_from_dict(rootdir)
    record.rootdir = rootdir
    segment_len = 16

    # configs from input modalities
    config_path = "/home/azureuser/AutonomousSystemsResearch/dataloader-benchmark/src/benchmarks/seqrecord/config.yaml"
    with open(config_path, mode="r") as f:
        import yaml

        config = yaml.safe_load(f)["inputs"]

    input_config = InputConfig()
    for key, modal in config.items():
        modal["kwargs"]["name"] = key
        input_config.add_input(modal)

    dl_config = {"num_workers": 6, "batch_size": 32, "prefetch_factor": 2}
    start_iter = perf_counter()
    test_iter(record, segment_len, dl_config, input_config)
    end_iter = perf_counter()
    print(f"{end_iter - start_iter =}")
