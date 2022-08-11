"""A TartanAIR dataset (both iterative and map style provide)
"""


from typing import List, Optional

import torch
from sklearn.utils import shuffle

from record import Record


# todo add map style
# todo: pre fetch and shuffle
# todo: transform (offline, online, to discuss with shuang)
# todo: add profile
class IterTartanAIRDataset(torch.utils.data.IterableDatasetDataset):
    """A torch dataset class that iteratively read from record files. It supports native pytorch multi-worker settings.

    # py lightning requires __next__ and a sampler
    Args:
        torch (_type_): _description_
    """

    def __init__(self, record: Record, segment_len: int, features: Optional[List[str]]) -> None:
        super().__init__()
        self.segmentproto = record.get_proto4segment(segment_len, features)
        self.record = record

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        itemidx4segment = self.segmentproto["itemidx4segment"]
        headidx4segment = self.segmentproto["headidx4segment"]
        segment_len = self.segmentproto["segment_len"]
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = itemidx4segment[headidx4segment[0]]["item_idx"]
            iter_end = itemidx4segment[headidx4segment[-1]]["item_idx"]
        else:  # in a worker process
            # split workload
            # ! drop last few samples
            per_worker = len(headidx4segment) // worker_info.num_workers
            # ! check worker id starts from zero
            worker_id = worker_info.id
            iter_start = itemidx4segment[headidx4segment[worker_id * per_worker]]
            iter_end = itemidx4segment[headidx4segment[(worker_id + 1) * per_worker - 1]]

        for item in self.record.decode_segment(iter_start, iter_end, segment_len):
            yield {key: torch.from_numpy(value) for key, value in item.items()}

    def __next__(self):
        # ! lightning seems to insist on implementing this function
        pass


if __name__ == "__main__":
    # sanity check dataloader
    rootdir = "/datadrive/azure_mounted_data/commondataset/tartanair-release1/abandonedfactory/records"
    record = Record.load_recordproto(rootdir)
    segment_len = 16
    tartanair_dataset = IterTartanAIRDataset(record, segment_len, features=None)  # read all features
    dataloader = torch.utils.data.Dataloader(
        tartanair_dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=True, prefetch_factor=2
    )
