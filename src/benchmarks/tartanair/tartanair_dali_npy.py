import argparse
import os
from distutils.util import strtobool

import mlflow
import numpy as np
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from src.benchmarks.benchmarker import Benchmarker
from src.benchmarks.common_opts import add_common_args
from src.benchmarks.tartanair.tartanair_opts import add_tartanair_args


@pipeline_def
def get_depth_image_pipeline(image_dir, device, random_shuffle, initial_fill, read_ahead):
    print("\n\n\nget_depth_image_pipeline():")
    data = fn.readers.numpy(
        device=device,
        file_root=image_dir,
        file_filter="*.npy",
        random_shuffle=random_shuffle,
        initial_fill=initial_fill,
        read_ahead=read_ahead,
        name="Reader",
    )
    return data


# ref: https://github.com/castacks/tartanair_tools/blob/master/TartanAir_Sample.ipynb
def depth2vis(depth, maxthresh=50):
    depthvis = np.clip(depth, 0, maxthresh)
    depthvis = depthvis / maxthresh * 255
    depthvis = depthvis.astype(np.uint8)
    depthvis = np.tile(depthvis.reshape(depthvis.shape + (1,)), (1, 1, 3))

    return depthvis


def visualize_depth_images(
    image_batch, batch_size, viz_method="depth2viz", is_amlt=False, result_num_cols=10, result_dir="debug_viz"
):
    print("\n\n\nvisualize_depth_images():")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    columns = result_num_cols
    rows = (batch_size + 1) // (columns)
    fig = plt.figure(figsize=(24, (24 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows * columns):
        plt.subplot(gs[j])
        plt.axis("off")
        if viz_method == "depth2viz":
            depth = image_batch.at(j)
            depthviz = depth2vis(depth)
            plt.imshow(depthviz)
        elif viz_method == "rgb":
            plt.imshow(image_batch.at(j))
        else:
            raise ValueError()

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if is_amlt:
        mlflow.log_figure(fig, os.path.join(result_dir, "image.png"))
    else:
        plt.savefig(os.path.join(result_dir, "image.png"))

    print("visualize_depth_images() done")


def visualize_depth_image_pipeline(args):
    print("\n\n\nvisualize_depth_image_pipeline():")
    pipe = get_depth_image_pipeline(
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        device=args.device,
        device_id=args.device_id,
        random_shuffle=args.random_shuffle,
        initial_fill=args.initial_fill,
        read_ahead=args.read_ahead,
        seed=args.seed,
    )
    pipe.build()
    pipe_out = pipe.run()
    (images,) = pipe_out
    visualize_depth_images(image_batch=images, batch_size=args.batch_size, is_amlt=args.is_amlt, result_dir="debug_viz")


def get_dataloader(args):
    pipe = get_depth_image_pipeline(
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        device=args.device,
        device_id=args.device_id,
        random_shuffle=args.random_shuffle,
        initial_fill=args.initial_fill,
        read_ahead=args.read_ahead,
        seed=args.seed,
    )

    dataloader = DALIGenericIterator(pipe, ["data"], reader_name="Reader")
    return dataloader


def benchmark(args):
    dataloader = get_dataloader(args)
    benchmarker = Benchmarker(verbose=args.verbose, dataset="tartanair", library="dali")
    benchmarker.set_dataloader(dataloader)
    benchmarker.benchmark_tartanair(args)


def add_dali_args(group):
    group.add_argument("--seed", type=int, default=42)
    group.add_argument("--num_threads", type=int, default=os.cpu_count())
    group.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "gpu"]
    )  # use gpu for GPUDirect Storage Support. needs cuda>=11.4
    group.add_argument("--device_id", type=int, default=0)
    group.add_argument("--random_shuffle", default="yes", type=lambda x: bool(strtobool(x)))
    group.add_argument("--initial_fill", type=int, default=100)
    group.add_argument(
        "--image_dir",
        type=str,
        default="/datadrive/localdatasets/tartanair-release1/abandonedfactory/Easy/P000/depth_left",
    )
    group.add_argument("--is_amlt", default="no", type=lambda x: bool(strtobool(x)))
    group.add_argument("--read_ahead", default="yes", type=lambda x: bool(strtobool(x)))
    group.add_argument(
        "--cache_header_information",
        default="yes",
        type=lambda x: bool(strtobool(x)),
        help="If set to True, the header information for each file is cached, improving access speed.",
    )


def main(args):
    benchmark(args)
    visualize_depth_image_pipeline(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    add_common_args(parser.add_argument_group("common args"))
    add_tartanair_args(parser.add_argument_group("tartanair args"))
    add_dali_args(parser.add_argument_group("dali args"))

    args = parser.parse_args()
    main(args)
