import argparse


def get_climate_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--use", type=str, default="pretrain", choices=["pretrain", "forecast"], help="Use forecast or pretrain"
    )
    parser.add_argument("--dataset_type", type=str, default="npy", choices=["npy", "zarr"], help="Dataset type")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/datadrive/localdatasets/climate/1.40625deg_monthly_np/val",
        help="Path to dataset",
    )
    parser.add_argument(
        "--variables",
        default=["z", "r", "u", "v", "t", "t2m", "u10", "v10"],
        help="list of ___ (strings)",
        nargs="+",
        type=str,
        choices=["z", "r", "u", "v", "t", "t2m", "u10", "v10"],
    )
    args = parser.parse_args()
    return args
