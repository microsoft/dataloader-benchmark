### dataloading speed benchmark

##### locally on vm
iterative style

with `segment_len=16` and all modalities from tartanair, we have
mins|num_workers| shuffle buffer size | prefetch factor|
|--- | --- | --- | ---|
|18| 4 | 100 | 2|

