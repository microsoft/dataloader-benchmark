from timeit import default_timer as timer

import mlflow


class Benchmarker:
    def __init__(self, verbose=False, dataset="tartanair", library="pytorch"):
        self.verbose = verbose
        self.dataset = dataset
        self.library = library

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

    def benchmark_tartanair(self, args):
        print(f"dataloader length: {len(self.dataloader)} batches")

        time_copy = 0.0
        start = timer()
        last = start
        num_batches = 0

        for batch_idx, batch in enumerate(self.dataloader):
            start_copy = timer()
            if self.library == "pytorch":
                for key, value in batch.items():
                    value.cuda()

            if self.library == "ffcv":
                for modality in batch:
                    modality.cuda()

            if self.library == "dali":
                batch[0]["data"].cuda()  # already on cuda, image_left only

            time_copy = time_copy + (timer() - start_copy)

            if batch_idx == 0:
                first = timer()

            if self.verbose:
                print(f"batch_idx {batch_idx} took {(timer() - last):.3f} seconds")
            last = timer()
            num_batches += 1

        last = timer()

        time_first_batch = first - start
        time_per_batch = (last - start) / num_batches
        time_per_batch_without_first = (last - first) / (num_batches - 1)
        time_copy_per_batch = time_copy / num_batches

        metrics = {}
        metrics["num_workers"] = args.num_workers
        metrics["batch_size"] = args.batch_size
        metrics["num_seq"] = args.num_seq
        metrics["seq_len"] = args.seq_len
        metrics["time_per_batch_without_first"] = time_per_batch_without_first
        metrics["time_per_batch"] = time_per_batch
        metrics["time_first_batch"] = time_first_batch
        metrics["time_copy_per_batch"] = time_copy_per_batch
        mlflow.log_metrics(metrics, step=0)

        with open(args.benchmark_results_file, "a") as f:
            f.write(
                f"{self.dataset}, "
                f"{self.library}, "
                f"{' '.join(args.modalities)}, "
                f"{args.train_transform}, "
                f"{args.batch_size}, "
                f"{args.num_workers}, "
                f"{args.num_seq}, "
                f"{args.seq_len}, "
                f"{time_first_batch:.3f}, "
                f"{time_per_batch:.3f}, "
                f"{time_per_batch_without_first:.3f}, "
                f"{time_copy_per_batch:.3f} "
                f"\n"
            )

        print(f"{time_first_batch:.3f} secs for the first batch")
        print(f"{time_per_batch:.3f} secs per batch")
        print(f"{time_per_batch_without_first:.3f} secs per batch without counting first batch")
        print(f"{time_copy_per_batch:.3f} secs per batch for copying from cpu to gpu")

    # todo clip_len / num_seq being used in mushr?
    # todo copy_to_gpu for mushr is untested
    def benchmark_mushr(self, args):
        print(f"dataloader length: {len(self.dataloader)} batches")

        time_copy = 0.0
        start = timer()
        last = start
        num_batches = 0

        for batch_idx, batch in enumerate(self.dataloader):
            start_copy = timer()
            for key, value in batch[0].items():
                value.cuda()
            batch[1].cuda()
            time_copy = time_copy + (timer() - start_copy)

            if batch_idx == 0:
                first = timer()

            if self.verbose:
                print(f"batch_idx {batch_idx} took {(timer() - last):.3f} seconds")

            last = timer()
            num_batches += 1

        last = timer()

        time_first_batch = first - start
        time_per_batch = (last - start) / num_batches
        time_per_batch_without_first = (last - first) / (num_batches - 1)
        time_copy_per_batch = time_copy / num_batches

        metrics = {}
        metrics["num_workers"] = args.num_workers
        metrics["batch_size"] = args.batch_size
        metrics["num_seq"] = args.num_seq
        metrics["seq_len"] = args.seq_len
        metrics["time_per_batch_without_first"] = time_per_batch_without_first
        metrics["time_per_batch"] = time_per_batch
        metrics["time_first_batch"] = time_first_batch
        metrics["time_copy_per_batch"] = time_copy_per_batch
        mlflow.log_metrics(metrics, step=0)

        with open(args.benchmark_results_file, "a") as f:
            f.write(
                f"{self.dataset}, "
                f"{self.library}, "
                f"{args.batch_size}, "
                f"{args.num_workers}, "
                f"{args.num_seq}, "
                f"{args.seq_len}, "
                f"{time_first_batch:.3f}, "
                f"{time_per_batch:.3f}, "
                f"{time_per_batch_without_first:.3f}, "
                f"{time_copy_per_batch:.3f}\n"
            )

        print(f"{time_first_batch:.3f} secs for the first batch")
        print(f"{time_per_batch:.3f} secs per batch")
        print(f"{time_per_batch_without_first:.3f} secs per batch without counting first batch")
        print(f"{time_copy_per_batch:.3f} secs per batch for copying from cpu to gpu")

    def benchmark_climate(self, args):
        time_copy = 0.0
        num_batches = 0
        start = timer()
        last = start
        for batch_idx, batch in enumerate(self.dataloader):
            start_copy = timer()
            if args.use == "forecast":
                x, y = batch[0].cuda(), batch[1].cuda()
            elif args.use == "pretrain":
                traj = batch[0].cuda()
            time_copy += timer() - start_copy

            if self.verbose:
                print(f"batch_idx {batch_idx} took {(timer() - last):.3f} seconds")
            last = timer()
            num_batches += 1

            if batch_idx == 0:
                first = timer()

        last = timer()

        time_copy_per_batch = time_copy / num_batches
        time_first_batch = first - start
        time_per_batch = (last - start) / num_batches
        time_per_batch_without_first = (last - first) / (num_batches - 1)

        metrics = {}
        metrics["num_workers"] = args.num_workers
        metrics["batch_size"] = args.batch_size
        metrics["time_per_batch_without_first"] = time_per_batch_without_first
        metrics["time_per_batch"] = time_per_batch
        metrics["time_first_batch"] = time_first_batch
        metrics["time_copy_per_batch"] = time_copy_per_batch
        mlflow.log_metrics(metrics, step=0)

        with open(args.benchmark_results_file, "a") as f:
            f.write(
                f"{self.dataset}, "
                f"{self.library}, "
                f"{args.use}, "
                f"{args.batch_size}, "
                f"{args.num_workers}, "
                f"{time_first_batch:.3f}, "
                f"{time_per_batch:.3f}, "
                f"{time_per_batch_without_first:.3f}, "
                f"{time_copy_per_batch:.3f}\n"
            )

        print(f"{time_first_batch:.3f} secs for the first batch")
        print(f"{time_per_batch:.3f} secs per batch")
        print(f"{time_per_batch_without_first:.3f} secs per batch without counting first batch")
        print(f"{time_copy_per_batch:.3f} secs per batch for copying from cpu to gpu")

    def benchmark_climate_dali(self, args):
        time_copy = 0.0
        num_batches = 0
        start = timer()
        last = start
        for batch_idx, batch in enumerate(self.dataloader):
            start_copy = timer()
            # already on gpu
            if args.use == "forecast":
                pass
            elif args.use == "pretrain":
                data = batch[0]["data"]
            time_copy += timer() - start_copy
            if batch_idx == 0:
                first = timer()

            if self.verbose:
                print(f"batch_idx {batch_idx} took {(timer() - last):.3f} seconds")

            last = timer()
            num_batches += 1

        last = timer()

        time_copy_per_batch = time_copy / num_batches
        time_first_batch = first - start
        time_per_batch = (last - start) / num_batches
        time_per_batch_without_first = (last - first) / (num_batches - 1)

        metrics = {}
        metrics["num_workers"] = args.num_workers
        metrics["batch_size"] = args.batch_size
        metrics["time_per_batch_without_first"] = time_per_batch_without_first
        metrics["time_per_batch"] = time_per_batch
        metrics["time_first_batch"] = time_first_batch
        metrics["time_copy_per_batch"] = time_copy_per_batch
        mlflow.log_metrics(metrics, step=0)

        with open(args.benchmark_results_file, "a") as f:
            f.write(
                f"{self.dataset}, "
                f"{self.library}, "
                f"{args.use}, "
                f"{args.batch_size}, "
                f"{args.num_workers}, "
                f"{time_first_batch:.3f}, "
                f"{time_per_batch:.3f}, "
                f"{time_per_batch_without_first:.3f}, "
                f"{time_copy_per_batch:.3f}\n"
            )

        print(f"{time_first_batch:.3f} secs for the first batch")
        print(f"{time_per_batch:.3f} secs per batch")
        print(f"{time_per_batch_without_first:.3f} secs per batch without counting first batch")
        print(f"{time_copy_per_batch:.3f} secs per batch for copying from cpu to gpu")
