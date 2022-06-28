from timeit import default_timer as timer

import mlflow

from data import build_loader
from utils.opts import parse_args


def get_tartan_dataset_and_loader(args):
    data_type = ["image_left"]
    if args.flow:
        data_type.append("flow_flow")
    train_dataset, _, train_loader, _, _ = build_loader(args, data_type)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    return train_dataset, train_loader


def benchmark(args):
    train_dataset, train_dataloader = get_tartan_dataset_and_loader(args)
    start = timer()
    last = start
    num_batches = len(train_dataloader)

    print(f"train_dataloader length {num_batches}")

    for batch_idx, _ in enumerate(train_dataloader):
        if batch_idx == 0:
            first = timer()
        # print(f"batch_idx {batch_idx} took {(timer() - last):.3f} seconds")
        # last = timer()

    last = timer()

    time_first_batch = first - start
    time_per_batch = (last - start) / num_batches
    time_per_batch_without_first = (last - first) / num_batches

    print(f"{time_per_batch:.3f} secs per batch")
    print(
        f"{time_per_batch_without_first:.3f} secs per batch without counting first batch"
    )
    print(f"{time_first_batch:.3f} secs for the first batch")

    mlflow.log_metric(key="num_workers", value=args.workers, step=0)
    mlflow.log_metric(key="batch_size", value=args.batch_size, step=0)
    mlflow.log_metric(key="num_seq", value=args.num_seq, step=0)
    mlflow.log_metric(key="seq_len", value=args.seq_len, step=0)
    mlflow.log_metric(
        key="time_per_batch_without_first", value=time_per_batch_without_first, step=0
    )
    mlflow.log_metric(key="time_per_batch", value=time_per_batch, step=0)
    mlflow.log_metric(key="time_first_batch", value=time_first_batch, step=0)

    with open(args.benchmark_results_file, "a") as f:
        f.write(
            f"{args.batch_size}, {args.workers}, {args.num_seq}, {args.seq_len}, {time_per_batch_without_first:.3f},  {time_per_batch:.3f}, {time_first_batch:.3f} \n"
        )


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("trace_" + str(p.step_num) + ".json")


def pytorch_profiler_schedule(args):
    from torch.profiler import ProfilerActivity, profile, schedule

    train_dataset, train_dataloader = get_tartan_dataset_and_loader(args)
    print(len(train_dataloader))
    with profile(
        activities=[ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=False,
        schedule=schedule(wait=1, warmup=1, active=2),
        on_trace_ready=trace_handler,
    ) as prof:
        for batch_idx, _ in enumerate(train_dataloader):
            print(f"batch_idx {batch_idx}")


def main(args):
    print("start")
    benchmark(args)
    # pytorch_profiler_schedule(args)


def cprofile(args):
    import cProfile
    import pstats

    cProfile.run("main(args)", f"{__file__}.profile")
    s = pstats.Stats(f"{__file__}.profile")
    s.strip_dirs()
    s.sort_stats("time").print_stats(10)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    # cprofile(args)
