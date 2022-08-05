"""A TFRecord style dataloader (both iterative and map style provide)
"""


import torch


class TartanAIRDataset(torch.utils.data.Dataset):
    """A torch dataset class that iteratively read from record files. It supports native pytorch multi-worker settings.

    Args:
        torch (_type_): _description_
    """

    def __init__(self) -> None:
        super().__init__()

    def __iter__(self):
        pass

    def __next__(self):
        pass
