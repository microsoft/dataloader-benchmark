import os
from distutils.util import strtobool

import mlflow
import numpy as np
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from tartanair_ops import get_tartanair_args

from src.benchmarks.benchmarker import Benchmarker
from src.benchmarks.common_opts import get_common_args


@pipeline_def
def get_frame_seq_pipe(seq_path, sequence_length, img_crop, img_dim, device="cpu", device_ops="cpu"):
    video = fn.readers.sequence(file_root=seq_path, sequence_length=sequence_length, device=device, name="Reader")
    # tartanair like crop
    pos_x = fn.random.uniform(range=(0.0, 1.0))
    pos_y = fn.random.uniform(range=(0.0, 1.0))

    images_crop = fn.crop(video, crop=(img_crop, img_crop), crop_pos_x=pos_x, crop_pos_y=pos_y, device=device_ops)
    images_resized = fn.resize(images_crop, size=[img_dim, img_dim], device=device_ops)

    return images_resized


def visualize_sequence(sequence, sequence_length, is_amlt=False, result_num_cols=10, result_dir="debug_viz"):
    import matplotlib.gridspec as gridspec
    from matplotlib import pyplot as plt

    columns = result_num_cols
    rows = (sequence_length + 1) // (columns)
    fig = plt.figure(figsize=(32, (16 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows * columns):
        plt.subplot(gs[j])
        plt.axis("off")
        plt.imshow(sequence[j])

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if is_amlt:
            mlflow.log_figure(fig, os.path.join(result_dir, f"sequence_{j:05}.png"))
        else:
            plt.savefig(os.path.join(result_dir, f"sequence_{j:05}.png"))


def visualize_seq_pipe(args):
    pipe = get_frame_seq_pipe(
        seq_path=args.seq_dir,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        device_id=args.device_id,
        device=args.device,
        device_ops=args.device_ops,
        seed=args.seed,
        img_crop=args.img_crop,
        img_dim=args.img_dim,
    )
    pipe.build()
    n_iter = 10

    for idx in range(n_iter):
        pipe_out = pipe.run()
        sequences_out = np.array(pipe_out[0][0])
        print(f"Iteration {idx} shape: {sequences_out.shape}")
        visualize_sequence(
            sequence=sequences_out,
            sequence_length=args.sequence_length,
            is_amlt=args.is_amlt,
            result_dir=f"debug_viz/iter_{idx:05}",
        )


def get_dataloader(args):
    pipe = get_frame_seq_pipe(
        seq_path=args.seq_dir,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        device_id=args.device_id,
        num_threads=args.num_threads,
        device=args.device,
        device_ops=args.device_ops,
        seed=args.seed,
        img_crop=args.img_crop,
        img_dim=args.img_dim,
        # device=args.device,
        # initial_fill=args.initial_fill,
        # random_shuffle=args.random_shuffle,
        # py_num_workers=args.num_threads,
        # prefetch_queue_depth={"cpu_size": num_threads, "gpu_size": 1},
    )

    dataloader = DALIGenericIterator(pipe, ["data"], reader_name="Reader")

    return dataloader


def benchmark(args):
    dataloader = get_dataloader(args)
    benchmarker = Benchmarker(verbose=args.verbose, dataset="tartanair", library="dali")
    benchmarker.set_dataloader(dataloader)
    benchmarker.benchmark_tartanair(args)


def get_dali_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_threads", type=int, default=os.cpu_count())
    # nvidia.dali.fn.readers.sequence¶ only supports cpu backend
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--device_ops", type=str, default="cpu")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--random_shuffle", default="yes", type=lambda x: bool(strtobool(x)))
    parser.add_argument("--sequence_length", type=int, default=16)
    parser.add_argument("--initial_fill", type=int, default=100)
    parser.add_argument("--seq_dir", type=str, default="/datadrive/localdatasets/tartanair-release1-dali")
    parser.add_argument("--is_amlt", default="yes", type=lambda x: bool(strtobool(x)))

    args = parser.parse_args()
    return args


def main(args):
    benchmark(args)
    visualize_seq_pipe(args)


if __name__ == "__main__":
    args = get_common_args()
    tartanair_args = get_tartanair_args()
    dali_args = get_dali_args()

    args.__dict__.update(tartanair_args.__dict__)
    args.__dict__.update(dali_args.__dict__)

    main(args)
