from timeit import default_timer as timer

import mlflow


class Benchmarker:
    def __init__(self, copy_to_gpu=True, verbose=False):
        self.copy_to_gpu = copy_to_gpu
        self.verbose = verbose

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

    def benchmark_tartanair(self, args):
        time_copy = 0.0
        start = timer()
        last = start
        num_batches = len(self.dataloader)
        num_batches = 0

        print(f"train_dataloader length {num_batches}")

        for batch_idx, batch in enumerate(self.dataloader):
            if self.copy_to_gpu:
                start_copy = timer()
                for key, value in batch[0].items():
                    value.cuda()
                batch[1].cuda()
                time_copy = time_copy + (timer() - start_copy)

            if batch_idx == 0:
                first = timer()

            print(f"batch_idx {batch_idx} took {(timer() - last):.3f} seconds")
            last = timer()
            num_batches += 1

        last = timer()

        time_first_batch = first - start
        time_per_batch = (last - start) / num_batches
        time_per_batch_without_first = (last - first) / (num_batches - 1)

        mlflow.log_metric(key="num_workers", value=args.workers, step=0)
        mlflow.log_metric(key="batch_size", value=args.batch_size, step=0)
        mlflow.log_metric(key="num_seq", value=args.num_seq, step=0)
        mlflow.log_metric(key="seq_len", value=args.seq_len, step=0)
        mlflow.log_metric(key="time_per_batch_without_first", value=time_per_batch_without_first, step=0)
        mlflow.log_metric(key="time_per_batch", value=time_per_batch, step=0)
        mlflow.log_metric(key="time_first_batch", value=time_first_batch, step=0)

        if not self.copy_to_gpu:
            with open(args.benchmark_results_file, "a") as f:
                f.write(
                    f"{' '.join(args.modalities)}, {args.train_transform}, {args.batch_size}, {args.workers}, {args.num_seq}, {args.seq_len}, {time_first_batch:.3f}, {time_per_batch:.3f}, {time_per_batch_without_first:.3f}\n"
                )

        else:
            time_copy_per_batch = time_copy / num_batches
            print(f"{time_copy_per_batch:.3f} secs per batch for copying from cpu to gpu")
            mlflow.log_metric(key="time_copy_per_batch", value=time_copy_per_batch, step=0)
            with open(args.benchmark_results_file, "a") as f:
                f.write(
                    f"{' '.join(args.modalities)}, {args.train_transform}, {args.batch_size}, {args.workers}, {args.num_seq}, {args.seq_len}, {time_first_batch:.3f}, {time_per_batch:.3f}, {time_per_batch_without_first:.3f}, {time_copy_per_batch:.3f}\n"
                )

        if self.verbose:
            print(f"{time_first_batch:.3f} secs for the first batch")
            print(f"{time_per_batch:.3f} secs per batch")
            print(f"{time_per_batch_without_first:.3f} secs per batch without counting first batch")
