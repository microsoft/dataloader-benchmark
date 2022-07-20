import glob
import os

import numpy as np
from tqdm import tqdm

DEFAULT_VARS = ["z", "r", "u", "v", "t", "t2m", "u10", "v10"]


def npz2npyconcat(file):
    data = np.load(file)
    newdata = []
    for key in DEFAULT_VARS:
        newdata.append(data[key])

    newdata = np.concatenate(newdata, axis=1)
    return newdata


def npz2npyforecast(file, predict_range=6):
    data = np.load(file)
    inputs = np.concatenate([data[k][0:-predict_range:predict_range] for k in DEFAULT_VARS], axis=1)
    outputs = np.concatenate([data[k][predict_range::predict_range] for k in DEFAULT_VARS], axis=1)
    return inputs, outputs


def convert(path, outdir, use):
    files = glob.glob(os.path.join(path, "*.npz"))
    os.makedirs(os.path.join(outdir, use), exist_ok=True)
    for file in tqdm(files):
        if use == "pretrain":
            newdata = npz2npyconcat(file)

            newfile = os.path.join(outdir, use, os.path.basename(file).replace(".npz", ".npy"))
            np.save(newfile, newdata)

        else:
            newdata_in, newdata_out = npz2npyforecast(file)
            fname = os.path.basename(file).replace(".npz", ".npy")
            newfile_in = os.path.join(outdir, use, f"input_{fname}")
            newfile_out = os.path.join(outdir, use, f"output_{fname}")
            np.save(newfile_in, newdata_in)
            np.save(newfile_out, newdata_out)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to npz files")
    parser.add_argument("outdir", help="Path to output npy files")
    parser.add_argument("--use", type=str, default="pretrain")
    return parser.parse_args()


def main():
    args = parse_args()
    convert(args.path, args.outdir, args.use)


if __name__ == "__main__":
    main()
