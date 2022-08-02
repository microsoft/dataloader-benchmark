import distutils
import os
from timeit import default_timer as timer

import mlflow
import numpy as np
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator


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


def test_pipe_single_batch(args):
    print("\n\n\ntest_pipe_single_batch():")
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
    batch = [np.array(pipe_out[0][sample_idx]) for sample_idx in range(args.batch_size)]
    for sample_idx, sample in enumerate(batch):
        print(
            f"sample {sample_idx:05}, shape: {sample.shape}, sample.min(): {sample.min()}, sample.max(): {sample.max()}, sample.mean(): {sample.mean()}"
        )


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


def benchmark_pipeline(args):
    print("\n\n\nbenchmark_pipeline():")
    start = timer()
    last = start

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

    dali_iter = DALIGenericIterator(pipe, ["data"], reader_name="Reader")

    for batch_idx, batch_list in enumerate(dali_iter):
        print(f"batch_idx: {batch_idx:05}")
        if batch_idx == 0:
            first = timer()
            print(f"batch_list[0]['data'].shape: {batch_list[0]['data'].shape}")
        print(f"{(timer() - last):.3f} secs for this batch")
        last = timer()

    last = timer()

    time_first_batch = first - start
    time_per_batch = (last - start) / (batch_idx + 1)
    time_per_batch_without_first = (last - first) / (batch_idx + 1)

    print(f"{time_first_batch:.3f} secs for the first batch")
    print(f"{time_per_batch:.3f} secs per batch")
    print(f"{time_per_batch_without_first:.3f} secs per batch without counting first batch")

    mlflow.log_metric(key="time_per_batch_without_first", value=time_per_batch_without_first, step=0)
    mlflow.log_metric(key="time_per_batch", value=time_per_batch, step=0)
    mlflow.log_metric(key="time_first_batch", value=time_first_batch, step=0)


def get_parsed_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_results_file", default="benchmark_results_tartanair.csv", type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_threads", type=int, default=os.cpu_count())
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "gpu"]
    )  # use gpu for GPUDirect Storage Support. needs cuda>=11.4
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--img_dim", type=int, default=224)
    parser.add_argument("--random_shuffle", default="yes", type=lambda x: bool(distutils.util.strtobool(x)))
    parser.add_argument("--img_crop", type=int, default=448)
    parser.add_argument("--initial_fill", type=int, default=100)
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/datadrive/localdatasets/tartanair-release1/abandonedfactory/Easy/P000/depth_left",
    )
    parser.add_argument("--is_amlt", default="no", type=lambda x: bool(distutils.util.strtobool(x)))
    parser.add_argument("--read_ahead", default="yes", type=lambda x: bool(distutils.util.strtobool(x)))
    parser.add_argument(
        "--cache_header_information",
        default="yes",
        type=lambda x: bool(distutils.util.strtobool(x)),
        help="If set to True, the header information for each file is cached, improving access speed.",
    )

    args = parser.parse_args()
    return args


def main():
    args = get_parsed_args()
    # test_pipe_single_batch(args)
    benchmark_pipeline(args)
    visualize_depth_image_pipeline(args)

    print("\n\n\n")


if __name__ == "__main__":
    main()
