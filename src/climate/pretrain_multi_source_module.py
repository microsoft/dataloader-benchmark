import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torchdata.datapipes as dp
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from datamodules import BOUNDARIES, VAR_LEVEL_TO_NAME_LEVEL

from .pretrain_iterdataset import (Forecast, IndividualForecastDataIter,
                                   NpyReader, ShuffleIterableDataset)


def collate_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])
    lead_times = torch.stack([batch[i][2] for i in range(len(batch))])
    variables = batch[0][3]
    out_variables = batch[0][4]
    region_info = batch[0][5]
    return (
        inp,
        out,
        lead_times,
        [VAR_LEVEL_TO_NAME_LEVEL[v] for v in variables],
        [VAR_LEVEL_TO_NAME_LEVEL[v] for v in out_variables],
        region_info
    )


class MultiSourceTrainDatasetModule(LightningDataModule):
    def __init__(
        self,
        dict_root_dirs: Dict,
        dict_start_idx: Dict,
        dict_end_idx: Dict,
        dict_buffer_sizes: Dict,
        dict_in_variables: Dict,
        dict_out_variables: Dict,
        dict_max_predict_ranges: Dict = {'mpi-esm': 28},
        dict_random_lead_time: Dict = {'mpi-esm': True},
        dict_hrs_each_step: Dict = {'mpi-esm': 6},
        dict_histories: Dict = {'mpi-esm': 3},
        dict_intervals: Dict = {'mpi-esm': 6},
        dict_subsamples: Dict = {'mpi-esm': 1},
        region: str = 'Global',
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        out_variables = {}
        for k, list_out in dict_out_variables.items():
            if list_out is not None:
                out_variables[k] = list_out
            else:
                out_variables[k] = dict_in_variables[k]
        self.hparams.dict_out_variables = out_variables

        self.dict_lister_trains = {
            k: list(dp.iter.FileLister(os.path.join(root_dir, "train"))) for k, root_dir in dict_root_dirs.items()
        }
        self.train_dataset_args = {
            k: {
                "max_predict_range": dict_max_predict_ranges[k],
                "random_lead_time": dict_random_lead_time[k],
                "hrs_each_step": dict_hrs_each_step[k],
                "history": dict_histories[k],
                "interval": dict_intervals[k],
                "subsample": dict_subsamples[k],
            } for k in dict_root_dirs.keys()
        }

        self.transforms = self.get_normalize()
        self.output_transforms = self.get_normalize(self.hparams.dict_out_variables)

        self.dict_data_train: Optional[Dict] = None

    def get_region_info(self, region):
        region = BOUNDARIES[region]
        lat_range = region['lat_range']
        lon_range = region['lon_range']
        lat, lon = self.get_lat_lon()
        lat = lat[::-1] # -90 to 90 from south (bottom) to north (top)
        h, w = len(lat), len(lon)
        lat_matrix = np.expand_dims(lat, axis=1).repeat(w, axis=1)
        lon_matrix = np.expand_dims(lon, axis=0).repeat(h, axis=0)
        valid_cells = (lat_matrix >= lat_range[0]) & (lat_matrix <= lat_range[1]) & (lon_matrix >= lon_range[0]) & (lon_matrix <= lon_range[1])
        h_ids, w_ids = np.nonzero(valid_cells)
        h_from, h_to = h_ids[0], h_ids[-1]
        w_from, w_to = w_ids[0], w_ids[-1]
        patch_idx = -1
        p = self.patch_size
        valid_patch_ids = []
        min_h, max_h = 1e5, -1e5
        min_w, max_w = 1e5, -1e5
        for i in range(0, h, p):
            for j in range(0, w, p):
                patch_idx += 1
                if (i >= h_from) & (i + p - 1 <= h_to) & (j >= w_from) & (j + p - 1 <= w_to):
                    valid_patch_ids.append(patch_idx)
                    min_h = min(min_h, i)
                    max_h = max(max_h, i + p - 1)
                    min_w = min(min_w, j)
                    max_w = max(max_w, j + p - 1)
        return {
            'patch_ids': valid_patch_ids,
            'min_h': min_h,
            'max_h': max_h,
            'min_w': min_w,
            'max_w': max_w
        }

    def get_normalize(self, dict_variables: Optional[Dict] = None):
        if dict_variables is None:
            dict_variables = self.hparams.dict_in_variables
        dict_transforms = {}
        for k in dict_variables.keys():
            root_dir = self.hparams.dict_root_dirs[k]
            variables = dict_variables[k]
            normalize_mean = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
            mean = []
            for var in variables:
                if var != "tp":
                    mean.append(normalize_mean[VAR_LEVEL_TO_NAME_LEVEL[var]])
                else:
                    mean.append(np.array([0.0]))
            normalize_mean = np.concatenate(mean)
            normalize_std = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))
            normalize_std = np.concatenate(
                [normalize_std[VAR_LEVEL_TO_NAME_LEVEL[var]] for var in variables]
            )
            dict_transforms[k] = transforms.Normalize(normalize_mean, normalize_std)
        return dict_transforms

    def get_lat_lon(self):
        # assume different data sources have the same lat and lon coverage
        lat = np.load(os.path.join(list(self.hparams.dict_root_dirs.values())[0], "lat.npy"))
        lon = np.load(os.path.join(list(self.hparams.dict_root_dirs.values())[0], "lon.npy"))
        return lat, lon

    def set_patch_size(self, p):
        self.patch_size = p

    def setup(self, stage: Optional[str] = None):
        region_info = self.get_region_info(self.hparams.region)
        # load datasets only if they're not loaded already
        if not self.dict_data_train:
            dict_data_train = {}
            for k in self.dict_lister_trains.keys():
                lister_train = self.dict_lister_trains[k]
                start_idx = self.hparams.dict_start_idx[k]
                end_idx = self.hparams.dict_end_idx[k]
                variables = self.hparams.dict_in_variables[k]
                out_variables = self.hparams.dict_out_variables[k]
                dataset_args = self.train_dataset_args[k]
                transforms = self.transforms[k]
                output_transforms = self.output_transforms[k]
                buffer_size = self.hparams.dict_buffer_sizes[k]
                dict_data_train[k] = ShuffleIterableDataset(
                    IndividualForecastDataIter(
                        Forecast(
                            NpyReader(
                                lister_train,
                                start_idx=start_idx,
                                end_idx=end_idx,
                                variables=variables,
                                out_variables=out_variables,
                                shuffle=True,
                                multi_dataset_training=True
                            ),
                            **dataset_args,
                        ),
                        transforms,
                        output_transforms,
                        region_info=region_info
                    ),
                    buffer_size,
                )
            self.dict_data_train = dict_data_train

    def train_dataloader(self):
        if not torch.distributed.is_initialized():
            raise NotImplementedError("Only support distributed training")
        else:
            node_rank = int(os.environ["NODE_RANK"])
            # Requires setting up our job yaml `env_defaults` to have this variable setup correctly
            num_nodes = os.environ.get("NODES", None)
            if num_nodes is not None:
                num_nodes = int(num_nodes)
                assert num_nodes == len(self.dict_data_train.keys())

            # TODO: figure out how to assert that number of datasets is the same as number of nodes
            for idx, k in enumerate(self.dict_data_train.keys()):
                if idx == node_rank:
                    data_train = self.dict_data_train[k]
                    break

        # This assumes that the number of datapoints are going to be the same for all datasets
        return DataLoader(
            data_train,
            batch_size=self.hparams.batch_size,            
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )


# dataset_type = 'forecast'
# dict_root_dirs = {
#     'mpi-esm': '/datadrive/datasets/CMIP6/MPI-ESM/5.625deg_equally_np_all_levels',
#     'taiesm': '/datadrive/datasets/CMIP6/TaiESM1/5.625deg_equally_np_all_levels'
# }
# dict_buffer_sizes = {'mpi-esm': 1000, 'taiesm': 1000}
# dict_in_variables = {
#     'mpi-esm': ['t2m', 'z_500', 't_850'],
#     'taiesm': ['z_500', 't_850']
# }
# dict_out_variables = {
#     'mpi-esm': ['z_500', 't_850'],
#     'taiesm': ['z_500', 't_850']
# }
# dict_predict_ranges = {'mpi-esm': 12, 'taiesm': 12}
# dict_histories = {'mpi-esm': 1, 'taiesm': 1}
# dict_intervals = {'mpi-esm': 0, 'taiesm': 0}
# dict_subsamples = {'mpi-esm': 1, 'taiesm': 1}

# datamodule = MultiSourceTrainDatasetModule(
#     dataset_type,
#     dict_root_dirs,
#     dict_buffer_sizes,
#     dict_in_variables,
#     dict_out_variables,
#     dict_predict_ranges=dict_predict_ranges,
#     dict_histories=dict_histories,
#     dict_intervals=dict_intervals,
#     dict_subsamples=dict_subsamples,
#     batch_size=16,
#     num_workers=1,
#     pin_memory=False
# )
# datamodule.setup()
# dataloader = datamodule.train_dataloader()
# for batch in dataloader:
#     for k in batch.keys():
#         print (k)
#         x1, y1, in1, out1 = batch[k]
#         print (x1.shape)
#         print (y1.shape)
#         print (in1)
#         print (out1)
#     break
