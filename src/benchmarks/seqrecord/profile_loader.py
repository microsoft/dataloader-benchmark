from time import perf_counter

from src.benchmarks.seqrecord.seqrecord import SeqRecord
from src.benchmarks.seqrecord.tartanloader import test_iter

rootdir = "/datadrive/azure_mounted_data/commondataset2/tartanair-release1/abandonedfactory/records"
record = SeqRecord.load_record_from_dict(rootdir)
record.rootdir = rootdir
segment_len = 16

dl_config = {"num_workers": 0, "batch_size": 32, "prefetch_factor": 2}
start_iter = perf_counter()
test_iter(record, segment_len, dl_config)
end_iter = perf_counter()
print(f"{end_iter - start_iter =}")
