import sys
from os.path import abspath, dirname

sys.path.append(dirname(dirname(abspath(__file__))))

NAME_TO_VAR = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "toa_incident_solar_radiation": "tisr",
    "total_precipitation": "tp",
    "land_sea_mask": "lsm",
    "orography": "orography",
    "lattitude": "lat2d",
    "geopotential": "z",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "temperature": "t",
    "relative_humidity": "r",
    "specific_humidity": "q",
}

VAR_TO_NAME = {v: k for k, v in NAME_TO_VAR.items()}

SINGLE_LEVEL_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_pressure",
    "toa_incident_solar_radiation",
    "total_precipitation",
    "land_sea_mask",
    "orography",
    "lattitude",
]
PRESSURE_LEVEL_VARS = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "relative_humidity",
    "specific_humidity",
]
ALL_LEVELS = [50, 250, 500, 600, 700, 850, 925, 1000]

NAME_LEVEL_TO_VAR_LEVEL = {}

for var in SINGLE_LEVEL_VARS:
    NAME_LEVEL_TO_VAR_LEVEL[var] = NAME_TO_VAR[var]

for var in PRESSURE_LEVEL_VARS:
    for l in ALL_LEVELS:
        NAME_LEVEL_TO_VAR_LEVEL[var + "_" + str(l)] = NAME_TO_VAR[var] + "_" + str(l)

VAR_LEVEL_TO_NAME_LEVEL = {v: k for k, v in NAME_LEVEL_TO_VAR_LEVEL.items()}

DEFAULT_PRESSURE_LEVELS = {
    "u": [1000, 850, 500],
    "v": [1000, 850, 500],
    "z": [1000, 850, 500, 50],
    "t": [850, 500],
    "r": [850, 500],
}

BOUNDARIES = {
    "NorthAmerica": {"lat_range": (15, 65), "lon_range": (220, 300)},  # 8x14
    "SouthAmerica": {"lat_range": (-55, 20), "lon_range": (270, 330)},  # 14x10
    "Europe": {"lat_range": (30, 65), "lon_range": (0, 40)},  # 6x8
    "SouthAsia": {"lat_range": (-15, 45), "lon_range": (25, 110)},  # 10, 14
    "EastAsia": {"lat_range": (5, 65), "lon_range": (70, 150)},  # 10, 12
    "Australia": {"lat_range": (-50, 10), "lon_range": (100, 180)},  # 10x14
    "Global": {"lat_range": (-90, 90), "lon_range": (0, 360)},  # 32, 64
}
