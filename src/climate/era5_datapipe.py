import numpy as np
import torch
import torchdata.datapipes as dp
import xarray as xr

NAME_MAP = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "geopotential": "z",
    "temperature": "t",
    "relative_humidity": "r",
}


class ERA5Npy(dp.iter.IterDataPipe):
    def __init__(self, dp: dp.iter.IterDataPipe, variables):
        super().__init__()
        self.dp = dp
        self.variables = variables

    def __iter__(self):
        for path in self.dp:
            data = np.load(path)
            yield {k: data[k] for k in self.variables}


class ERA5Zarr(dp.iter.IterDataPipe):
    def __init__(self, dp: dp.iter.IterDataPipe, variables):
        super().__init__()
        self.dp = dp
        self.variables = variables

    def __iter__(self):
        for path in self.dp:
            data = xr.open_zarr(path)
            yield {k: data[k].to_numpy() for k in self.variables}


class ERA5Forecast(dp.iter.IterDataPipe):
    def __init__(self, dp: ERA5Npy, predict_range: int = 6) -> None:
        super().__init__()
        self.dp = dp
        self.predict_range = predict_range

    def __iter__(self):
        # TODO: this would not get us stuff across the years
        # i.e. where inputs are from previous years and output from next
        for data in self.dp:

            inputs = np.concatenate(
                [data[k][0 : -self.predict_range : self.predict_range] for k in data.keys()],
                axis=1,
            )
            outputs = np.concatenate(
                [data[k][self.predict_range :: self.predict_range] for k in data.keys()],
                axis=1,
            )
            yield torch.from_numpy(inputs), torch.from_numpy(outputs)


class IndividualDataIter(dp.iter.IterDataPipe):
    def __init__(self, dp: ERA5Forecast):
        super().__init__()
        self.dp = dp

    def __iter__(self):
        for (inp, out) in self.dp:
            assert inp.shape[0] == out.shape[0]
            for i in range(inp.shape[0]):
                # TODO: should we unsqueeze the first dimension?
                yield inp[i], out[i]
