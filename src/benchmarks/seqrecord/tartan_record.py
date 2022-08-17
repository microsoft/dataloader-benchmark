from logging import root
from time import perf_counter

import torch
from src.benchmarks.seqrecord.seqrecord import SeqRecord
from src.benchmarks.seqrecord.tartanloader import build_itertartanair_datapipe
from tqdm import tqdm


def test_iter(record, segment_len, dl_config):
    datapipe = build_itertartanair_datapipe(record, segment_len, dl_config)
    dataloader = torch.utils.data.DataLoader(
        datapipe, shuffle=True, num_workers=dl_config["num_workers"], prefetch_factor=dl_config["prefetch_factor"]
    )
    for batch in tqdm(dataloader):
        # is this the best way? we should do one-pass to 'small network'
        for key in batch:
            batch[key].cuda()
        torch.cuda.synchronize()


def main():
    storage_dir = ["/mnt/data/", "/datadrive/azure_mounted_data/commondataset2"]
    rootdir = storage_dir[0] + "/tartanair-release1/abandonedfactory/records"
    record = SeqRecord.load_recordobj(rootdir)
    record.rootdir = rootdir
    segment_len = 16

    dl_config = {"num_workers": 4, "batch_size": 32, "prefetch_factor": 2}
    start_iter = perf_counter()
    test_iter(record, segment_len, dl_config)
    end_iter = perf_counter()
    print(f"{end_iter - start_iter =}")
    # start_map = perf_counter()
    # test_map(record, segment_len, dl_config)
    # end_map = perf_counter()
    # print(f"{end_map - start_map =}")


if __name__ == "__main__":
    main()
