import os

import numpy as np


# in: '/datadrive/localdatasets/climate/1.40625deg_monthly_np/val/2013_12.npz'
# out: '12'
def get_month_from_npz_full_path(npz_path):
    return int(npz_path.split("/")[-1].split(".npz")[0].split("_")[-1])


DEFAULT_VARS = ["z", "r", "u", "v", "t", "t2m", "u10", "v10"]


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

npz_monthly_list = [
    os.path.join(dataset_root, monthly_root, split, fname)
    for fname in os.listdir(os.path.join(dataset_root, monthly_root, split))
]

# [1, 10, 11, 12, 2, 3, 4, 5, 6, 7, 8, 9]
# months = [get_month_from_npz_full_path(npz_path) for npz_path in npz_monthly_list]
npz_monthly_list.sort(key=lambda x: get_month_from_npz_full_path(x))

# sort accoring to month as we are missing leading zero in month idx

num_hours_cum = 0
for fname in npz_monthly_list:
    npz = np.load(fname)
    keys = list(npz.iterkeys())

    var_dim_cum = 0
    hour_dim = 0

    # for var in list(npz.iterkeys()):
    for var in DEFAULT_VARS:
        npz_monthly_var = npz[var]
        print(f"{fname}, {var:5} {VAR_TO_NAME[var]:30} shape: {npz_monthly_var.shape}")

        hour_dim, var_dim, _, _ = npz_monthly_var.shape
        VAR_TO_DIM[var] = var_dim

        # get corrsponding slice from yearly data
        npy_yearly_slice = npy_yearly[
            num_hours_cum : num_hours_cum + hour_dim, var_dim_cum : var_dim_cum + var_dim, :, :
        ]

        # check if equal
        assert npy_yearly_slice.shape == npz_monthly_var.shape
        np.testing.assert_array_equal(npy_yearly_slice, npz_monthly_var)
        # print(f"diff b/w npy_yearly_slice and npz_monthly_var: {(npy_yearly_slice-npz_monthly_var).mean()}")
        print("\n")

        var_dim_cum += var_dim

    num_hours_cum += hour_dim


print(num_hours_cum)
print(f"yearly.shape: {npy_yearly.shape}")
from pprint import pprint

pprint(VAR_TO_DIM)
