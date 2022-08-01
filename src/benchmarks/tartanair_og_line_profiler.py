from src.data.tartanair import build_loader
from src.utils.opts import parse_args


def get_tartanair_dataset(args):
    train_dataset, _, train_loader, _, _ = build_loader(args)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    return train_dataset, train_loader


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("trace_" + str(p.step_num) + ".json")


def pytorch_profiler_schedule(args):
    from torch.profiler import ProfilerActivity, profile, schedule

    train_dataset, train_dataloader = get_tartanair_dataset(args)
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
    pytorch_profiler_schedule(args)


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
    cprofile(args)
