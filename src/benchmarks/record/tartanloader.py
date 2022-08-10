"""A TFRecord style dataloader for TartanAIR dataset (both iterative and map style provide)
"""


from typing import List

import torch

from record import Record


# todo add map style
class IterTartanAIRDataset(torch.utils.data.IterableDatasetDataset):
    """A torch dataset class that iteratively read from record files. It supports native pytorch multi-worker settings.

    # py lightning requires __next__ and a sampler
    Args:
        torch (_type_): _description_
    """

    def __init__(self, record: Record, segment_len: int, features: List[str]) -> None:
        super().__init__()
        self.segmentproto = record.get_proto4segment(segment_len, features)
        self.record = record

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        item_idx_proto = self.segmentproto["item_idx_proto"]
        head_idx_proto = self.segmentproto["head_idx_proto"]
        segment_len = self.segmentproto["segment_len"]
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = item_idx_proto[head_idx_proto[0]]["item_idx"]
            iter_end = item_idx_proto[head_idx_proto[-1]]["item_idx"]
        else:  # in a worker process
            # split workload
            # ! drop last few samples
            per_worker = len(head_idx_proto) // worker_info.num_workers
            # ! check worker id starts from zero
            worker_id = worker_info.id
            iter_start = item_idx_proto[head_idx_proto[worker_id * per_worker]]
            iter_end = item_idx_proto[head_idx_proto[(worker_id + 1) * per_worker]]
        return iter(self.record.decode_segment(iter_start, iter_end, segment_len))

    def __next__(self):
        # ! lightning seems to insist on implementing this function
        pass
