import csv
import os
from timeit import default_timer as timer

import mlflow


class Benchmarker:
    def __init__(self, verbose=False, dataset="tartanair", library="pytorch", init_time=timer()):
        self.verbose = verbose
        self.dataset = dataset
        self.library = library
        self.init_time = init_time  # start time before calling dataloader

        self.datasets_list = ["tartanair", "climate", "mushr"]
        self.library_list = ["pytorch", "ffcv", "dali"]

        if self.dataset not in self.datasets_list:
            raise ValueError(
                f"Benchmarker: {self.dataset} is not supported. List of supported datasets: {self.datasets_list}"
            )

        if self.library not in self.library_list:
            raise ValueError(
                f"Benchmarker: {self.library} is not supported. List of supported libraries: {self.library_list}"
            )

        self.common_params_list = [
            "dataset",
            "library",
            "batch_size",
            "num_workers",
        ]

        self.common_metrics_list = [
            "time_per_batch_without_first",
            "time_per_batch_with_first",
            "data_prep_time",
            "time_first_batch",
            "time_copy_per_batch",
        ]

        # initialize empty metrics registry (floats only)
        self.metrics_registry = {}
        self.metrics_registry.update(dict.fromkeys(self.datasets_list))
        self.metrics_registry.update(dict.fromkeys(self.library_list))

        # initialize empty params registry (any type is allowed, strings preferably)
        self.params_registry = {}
        self.params_registry.update(dict.fromkeys(self.datasets_list))
        self.params_registry.update(dict.fromkeys(self.library_list))

        # define custom params for supported datasets
        self.params_registry["tartanair"] = dict.fromkeys(["modalities", "transform", "num_seq", "seq_len"])
        self.params_registry["climate"] = dict.fromkeys(["use"])
        self.params_registry["mushr"] = dict.fromkeys(["num_seq", "seq_len"])

        # define custom params for supported libraries
        self.params_registry["pytorch"] = dict.fromkeys([])
        self.params_registry["ffcv"] = dict.fromkeys(["order", "os_cache"])
        self.params_registry["dali"] = dict.fromkeys([])

        # define custom metrics for supported datasets
        self.metrics_registry["tartanair"] = dict.fromkeys([])
        self.metrics_registry["climate"] = dict.fromkeys([])
        self.metrics_registry["mushr"] = dict.fromkeys([])

        # define custom metrics for supported libraries
        self.metrics_registry["pytorch"] = dict.fromkeys([])
        self.metrics_registry["ffcv"] = dict.fromkeys([])
        self.metrics_registry["dali"] = dict.fromkeys([])

        # update metrics and params for this instance given dataset and library
        # not these are different dicts from the registry dicts

        # initiliaze dicts with common
        self.metrics = dict.fromkeys(self.common_metrics_list)
        self.params = dict.fromkeys(self.common_params_list)

        # intialize additional keys from custom dataset and library
        self.params.update(self.params_registry[self.dataset])
        self.params.update(self.params_registry[self.library])
        self.metrics.update(self.metrics_registry[self.dataset])
        self.metrics.update(self.metrics_registry[self.library])

        # set dataset and library param
        self.params["dataset"] = self.dataset
        self.params["library"] = self.library

        print(
            f"\n"
            f"Benchmarker initialized with dataset: {self.dataset}, library: {self.library}, and the following metrics:\n"
            f"{list(self.metrics.keys())} \n"
            f"and the following params:\n"
            f"{list(self.params.keys())}"
        )

        # CamelCase to snake_case convertor
        # camelcase2snakecase = re.compile(r"(?<!^)(?=[A-Z])")
        # self.metrics_vars = {key: camelcase2snakecase.sub("_", key).lower() for key, value in self.metrics.items()}

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

    def log_results(self):
        mlflow.log_params(self.params)
        mlflow.log_metrics(self.metrics, step=0)

        # write to csv
        # make headers as (params, metrics).
        fieldnames = list(self.params.keys())
        fieldnames.extend(list(self.metrics.keys()))
        file_exists = os.path.isfile(self.benchmark_results_file)

        result = {**self.params, **self.metrics}

        with open(self.benchmark_results_file, "a") as f:
            writer = csv.DictWriter(f, delimiter=",", lineterminator="\n", fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(result)

        for (key, value) in result.items():
            print(f"{key}: {value}")

    def benchmark_tartanair(self, args):
        print(f"dataloader length: {len(self.dataloader)} batches.\n")

        self.benchmark_results_file = args.benchmark_results_file
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

            # torch.cuda.synchronize()
            time_copy = time_copy + (timer() - start_copy)

            if batch_idx == 0:
                first = timer()

            if self.verbose:
                print(f"batch_idx {batch_idx} took {(timer() - last):.3f} seconds")
            last = timer()
            num_batches += 1

        last = timer()

        data_prep_time = start - self.init_time
        time_first_batch = first - start
        time_per_batch_with_first = (last - start) / num_batches
        time_per_batch_without_first = (last - first) / (num_batches - 1)
        time_copy_per_batch = time_copy / num_batches

        # common
        self.params["num_workers"] = args.num_workers
        self.params["batch_size"] = args.batch_size
        self.metrics["time_per_batch_without_first"] = round(time_per_batch_without_first, 3)
        self.metrics["time_per_batch_with_first"] = round(time_per_batch_with_first, 3)
        self.metrics["data_prep_time"] = round(data_prep_time, 3)
        self.metrics["time_first_batch"] = round(time_first_batch, 3)
        self.metrics["time_copy_per_batch"] = round(time_copy_per_batch, 3)

        # custom
        self.params["transform"] = args.train_transform
        self.params["modalities"] = " ".join(args.modalities)
        self.params["num_seq"] = args.num_seq
        self.params["seq_len"] = args.seq_len

        self.log_results()

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

        data_prep_time = start - self.init_time
        time_first_batch = first - start
        time_per_batch_with_first = (last - start) / num_batches
        time_per_batch_without_first = (last - first) / (num_batches - 1)
        time_copy_per_batch = time_copy / num_batches

        # common
        self.params["num_workers"] = args.num_workers
        self.params["batch_size"] = args.batch_size
        self.metrics["time_per_batch_without_first"] = round(time_per_batch_without_first, 3)
        self.metrics["time_per_batch_with_first"] = round(time_per_batch_with_first, 3)
        self.metrics["data_prep_time"] = round(data_prep_time, 3)
        self.metrics["time_first_batch"] = round(time_first_batch, 3)
        self.metrics["time_copy_per_batch"] = round(time_copy_per_batch, 3)

        # custom
        self.params["num_seq"] = args.num_seq
        self.params["seq_len"] = args.seq_len

        self.log_results()

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
        data_prep_time = start - self.init_time
        time_first_batch = first - start
        time_per_batch_with_first = (last - start) / num_batches
        time_per_batch_without_first = (last - first) / (num_batches - 1)

        # commons
        self.params["num_workers"] = args.num_workers
        self.params["batch_size"] = args.batch_size
        self.metrics["time_per_batch_without_first"] = round(time_per_batch_without_first, 3)
        self.metrics["time_per_batch_with_first"] = round(time_per_batch_with_first, 3)
        self.metrics["data_prep_time"] = round(data_prep_time, 3)
        self.metrics["time_first_batch"] = round(time_first_batch, 3)
        self.metrics["time_copy_per_batch"] = round(time_copy_per_batch, 3)

        # custom
        self.params["use"] = args.use

        self.log_results()

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
        data_prep_time = start - self.init_time
        time_first_batch = first - start
        time_per_batch_with_first = (last - start) / num_batches
        time_per_batch_without_first = (last - first) / (num_batches - 1)

        # commons
        self.params["num_workers"] = args.num_workers
        self.params["batch_size"] = args.batch_size
        self.metrics["time_per_batch_without_first"] = round(time_per_batch_without_first, 3)
        self.metrics["time_per_batch_with_first"] = round(time_per_batch_with_first, 3)
        self.metrics["data_prep_time"] = round(data_prep_time, 3)
        self.metrics["time_first_batch"] = round(time_first_batch, 3)
        self.metrics["time_copy_per_batch"] = round(time_copy_per_batch, 3)

        # custom
        self.params["use"] = args.use

        self.log_results()
