from tartanair_ops import get_tartanair_args

from src.benchmarks.common_opts import get_common_args
from src.data.tartanair import build_loader


def get_dataloader(args):
    _, _, train_loader, _, _ = build_loader(args)
    return train_loader


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("trace_" + str(p.step_num) + ".json")


def pytorch_profiler_schedule(args):
    from torch.profiler import ProfilerActivity, profile, schedule

    dataloader = get_dataloader(args)
    print(len(dataloader))
    with profile(
        activities=[ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=False,
        schedule=schedule(wait=1, warmup=1, active=2),
        on_trace_ready=trace_handler,
    ) as prof:
        for batch_idx, _ in enumerate(dataloader):
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
    args = get_common_args()
    args_tartanair = get_tartanair_args()
    args.__dict__.update(args_tartanair.__dict__)
    main(args)
    cprofile(args)
