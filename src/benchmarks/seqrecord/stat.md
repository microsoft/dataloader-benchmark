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

ffcv on tartanair with `sqe_len=1`,
| min | batch_size | num_workers | shuffle? | prefetch factor |
| --- | --- | ---| ---|
|15 | 32 | 4| random | NA|

offcv on tartanair with `sqe_len=16`, has memory issues.

### aml with amulet
data download and caching is unknonw.

original tartanair dataset
|dataloader|mins | batch_size| num_workers | shuffle (buffer size)? | prefetch factor|
|---|---| ---| ---| ---| ---|
|tartan og| 6*60 |32| 4| True | 1|
|tartan og| 350 |32| 4| False | 1|
|seqrecord | 18| 32 | 4 | 100| 2|


ffcv with `seg_len=1`
|dataloader|mins | batch_size| num_workers | shuffle (size)? | prefetch factor|
|---|---| ---| ---| ---| ---|
|ffcv| 10 |32| 4| quasi-random | -|

### local vm with blobfile 

|dataloader|mins | batch_size| num_workers | shuffle (size)? | prefetch factor|
|---|---| ---| ---| ---| ---|
|seqrecord | 45| 32 | 4 | 100| 2|