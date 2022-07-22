import os

import numpy as np

NAME_TO_VAR = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "geopotential": "z",
    "temperature": "t",
    "relative_humidity": "r",
}

VAR_TO_NAME = {v: k for k, v in NAME_TO_VAR.items()}

VAR_TO_DIM = {}

dataset_root = "/datadrive/localdatasets/climate"
monthly_root = "1.40625deg_monthly_np"
split = "val"
fname_year_npy = "trial_npy/pretrain/2013.npy"
# (8760, 17, 128, 256)
npy_yearly = np.load(os.path.join(dataset_root, fname_year_npy))
print(f"yearly.shape: {npy_yearly.shape}")

npz_monthly_list = [
    os.path.join(dataset_root, monthly_root, split, fname)
    for fname in os.listdir(os.path.join(dataset_root, monthly_root, split))
]

cumulative_leading_dim = 0
for fname in npz_monthly_list:
    npz = np.load(fname)
    keys = list(npz.iterkeys())
    cumulative_leading_dim += npz[keys[0]].shape[0]
    for key in list(npz.iterkeys()):
        print(f"{fname}, {key:5} {VAR_TO_NAME[key]:20} shape: {npz[key].shape}")
        VAR_TO_DIM[key] = npz[key].shape[1]
bla = 1

print(cumulative_leading_dim)
from pprint import pprint

pprint(VAR_TO_DIM)
