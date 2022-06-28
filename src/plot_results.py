import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("benchmark_results.csv")

inputs = ["batch_size", "num_cpu", "num_seq", "seq_len"]
outputs = [
    "time_per_batch_without_first_batch",
    "time_per_batch",
    "time_for_first_batch",
]

# batch_sizes = df[input].unique()
# batch_sizes.sort()
df.groupby(["batch_size", "num_seq", "seq_len", "num_cpu"])

# df.plot(kind="hist", x="batch_size", y="time_per_batch_without_first_batch")
# plt.savefig("result_1.png")
