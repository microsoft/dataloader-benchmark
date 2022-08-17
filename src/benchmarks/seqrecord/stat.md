## dataloading speed benchmark

### locally on vm 
data access is taken care of by blobfuse

iterative style

with `segment_len=16` and all modalities from tartanair, we have
mins|batch_size|num_workers| shuffle buffer size | prefetch factor|
|--- |---| --- | --- | ---|
|18| 8|4 | 100 | 2|

map style without shuffling (data are essentially sequentially read)

mins|batch_size|num_workers| shuffle? | prefetch factor|
|--- | ---|--- | --- | ---|
|32| 8|4 | False | 2|


original tartanair dataset
|mins | batch_size|num_workers | shuffle?| prefetch factor|
|--- |---|--- | ---| ---|
|  44   |8|4 | False| 1|
|  50   |8| 4| True | 1|

### aml with amulet

original tartanair dataset
|dataloader|mins | batch_size| num_workers | shuffle (size)? | prefetch factor|
|---|---| ---| ---| ---| ---|
|tartan og| 6*60 |32| 4| True | 1|
|seqrecord | 18| 32 | 4 | 1000| 2|