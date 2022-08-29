import argparse
import os
from distutils.util import strtobool

import mlflow
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from src.benchmarks.benchmarker import Benchmarker
from src.benchmarks.common_opts import add_common_args
from src.benchmarks.tartanair.tartanair_opts import add_tartanair_args


@pipeline_def
def get_rgb_image_pipeline(image_dir, random_shuffle, initial_fill, device, img_crop, img_dim):
    print("\n\n\nget_rgb_image_pipeline():")
    pngs, labels = fn.readers.file(
        file_root=image_dir, random_shuffle=random_shuffle, initial_fill=initial_fill, name="Reader"
    )

    # no crop
    # images = fn.decoders.image(pngs, device=device)

    # tartanair like crop
    pos_x = fn.random.uniform(range=(0.0, 1.0))
    pos_y = fn.random.uniform(range=(0.0, 1.0))

    images_crop = fn.decoders.image_crop(
        pngs, crop=(img_crop, img_crop), crop_pos_x=pos_x, crop_pos_y=pos_y, device=device
    )
    images_resized = fn.resize(images_crop, size=[img_dim, img_dim])

    return images_resized


def visualize_rgb_images(image_batch, batch_size, is_amlt=False, result_num_cols=10, result_dir="debug_viz"):
    print("\n\n\nvisualize_rgb_images():")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    columns = result_num_cols
    rows = (batch_size + 1) // (columns)
    fig = plt.figure(figsize=(24, (24 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows * columns):
        plt.subplot(gs[j])
        plt.axis("off")
        plt.imshow(image_batch.at(j))

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if is_amlt:
        mlflow.log_figure(fig, os.path.join(result_dir, "image.png"))
    else:
        fig.savefig(os.path.join(result_dir, "image.png"))

    print("visualize_rgb_images() done")


def visualize_rgb_image_pipeline(args):
    print("\n\n\nvisualize_rgb_image_pipeline():")
    pipe = get_rgb_image_pipeline(
        image_dir=args.image_dir,
        random_shuffle=args.random_shuffle,
        initial_fill=args.initial_fill,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        device_id=args.device_id,
        device=args.device,
        seed=args.seed,
        img_crop=args.img_crop,
        img_dim=args.img_dim,
    )
    pipe.build()
    pipe_out = pipe.run()
    # print(pipe_out)
    (images,) = pipe_out
    # print("Images is_dense_tensor: " + str(images.is_dense_tensor()))
    # print("Images is_dense_tensor: " + str(images.is_dense_tensor()))
    # print(labels)
    visualize_rgb_images(
        image_batch=images.as_cpu(), batch_size=args.batch_size, is_amlt=args.is_amlt, result_dir="debug_viz"
    )


# https://docs.nvidia.com/deeplearning/dali/user-guide/docs/advanced_topics_performance_tuning.html#prefetching-queue-depth
def performance_tuning(args):
    print("\n\n\nperformance_tuning():")
    pipe = get_rgb_image_pipeline(
        image_dir=args.image_dir,
        random_shuffle=args.random_shuffle,
        initial_fill=args.initial_fill,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        device_id=args.device_id,
        device=args.device,
        seed=args.seed,
        img_crop=args.img_crop,
        img_dim=args.img_dim,
        enable_memory_stats=True,
    )
    pipe.build()
    pipe.run()
    stats = Pipeline.executor_statistics(pipe)
    from pprint import pprint

    pprint(stats)
    print(stats["max_reserved_memory_size"])


def get_dataloader(args):
    pipe = get_rgb_image_pipeline(
        batch_size=args.batch_size,
        device_id=args.device_id,
        device=args.device,
        image_dir=args.image_dir,
        initial_fill=args.initial_fill,
        num_threads=args.num_threads,
        random_shuffle=args.random_shuffle,
        img_crop=args.img_crop,
        img_dim=args.img_dim,
        seed=args.seed,
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


def add_dali_args(group):
    group.add_argument("--seed", type=int, default=42)
    group.add_argument("--num_threads", type=int, default=os.cpu_count())
    group.add_argument("--device", type=str, default="mixed")
    group.add_argument("--device_id", type=int, default=0)
    group.add_argument("--random_shuffle", default="yes", type=lambda x: bool(strtobool(x)))
    group.add_argument("--initial_fill", type=int, default=100)
    group.add_argument("--image_dir", type=str, default="/datadrive/localdatasets/tartanair-release1-dali/")
    group.add_argument("--is_amlt", default="yes", type=lambda x: bool(strtobool(x)))


def main(args):
    benchmark(args)
    visualize_rgb_image_pipeline(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    add_common_args(parser.add_argument_group(title="common args"))
    add_tartanair_args(parser.add_argument_group(title="tartanair args"))
    add_dali_args(parser.add_argument_group(title="dali args"))

    args = parser.parse_args()

    main(args)
