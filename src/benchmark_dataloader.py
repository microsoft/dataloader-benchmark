import time

from data import build_loader
from utils.opts import parse_args


def get_tartan_dataset_and_loader(args):
    data_type = ["image_left"]
    if args.flow:
        data_type.append("flow_flow")
    print("--used modalities are:", data_type)
    train_dataset, _, train_loader, _, _ = build_loader(args, data_type)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    return train_dataset, train_loader


def benchmark(args):
    train_dataset, train_dataloader = get_tartan_dataset_and_loader(args)
    tic = time.time()
    for batch_idx, _ in enumerate(train_dataloader):
        print("batch_idx {batch_idx}")
    print(f"time taken for {len(train_dataloader)} batches: {time.time()-tic} seconds")


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("trace_" + str(p.step_num) + ".json")


def pytorch_profiler_schedule(args):
    from torch.profiler import ProfilerActivity, profile, schedule

    train_dataset, train_dataloader = get_tartan_dataset_and_loader(args)
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
    # benchmark(args)
    pytorch_profiler_schedule(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
