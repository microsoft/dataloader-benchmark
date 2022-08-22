import os
from time import perf_counter

import blobfile as bf
import torch
from src.benchmarks.seqrecord.seqrecord import SeqRecord, get_azure_url
from src.benchmarks.seqrecord.tartanloader import build_itertartanair_datapipe
from tqdm import tqdm


def test_iter(record, segment_len, dl_config, azure_dir):
    datapipe = build_itertartanair_datapipe(record, segment_len, dl_config, azure_dir)
    dataloader = torch.utils.data.DataLoader(
        datapipe, shuffle=True, num_workers=dl_config["num_workers"], prefetch_factor=dl_config["prefetch_factor"]
    )
    for batch in tqdm(dataloader):
        # is this the best way? we should do one-pass to 'small network'
        for key in batch:
            batch[key].cuda()
        torch.cuda.synchronize()


def main(config_path: str):
    import yaml

    with open(config_path, mode="r") as f:
        config = yaml.safe_load(f)
    os.environ["AZURE_STORAGE_KEY"] = config["azure_key"]
    storage_dir = get_azure_url(config["account"], config["container"])
    rootdir = bf.join(storage_dir, "tartanair-release1/abandonedfactory/records")
    record = SeqRecord.load_record_from_dict(rootdir, is_azure_blob_dir=True)
    record.rootdir = rootdir
    segment_len = 16

    dl_config = {"num_workers": 4, "batch_size": 32, "prefetch_factor": 2}
    start_iter = perf_counter()
    test_iter(record, segment_len, dl_config, rootdir)
    end_iter = perf_counter()
    print(f"{end_iter - start_iter =}")
    # start_map = perf_counter()
    # test_map(record, segment_len, dl_config)
    # end_map = perf_counter()
    # print(f"{end_map - start_map =}")


if __name__ == "__main__":
    config_path = "./src/benchmarks/seqrecord/config.yaml"
    main(config_path)
