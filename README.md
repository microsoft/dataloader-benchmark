# dataloader-benchmark

## Benchmarking dataloader

### local
- generate smaller json by hand :

  - copy first n lines:
    ```
    head -17411 /datadrive/commondatasets/tartanair-release1/train_ann_debug.json  > /datadrive/commondatasets/tartanair-release1/train_ann_debug_ratnesh.json
    head -810 /datadrive/commondatasets/tartanair-release1/train_ann_debug.json  > /datadrive/commondatasets/tartanair-release1/train_ann_debug_ratnesh_100_frames.json
    ```
  - add some closing braces manually

- generate smaller json with a script

- Usage

  ```
  python src/benchmarks/tartanair_og.py  -h
  ```

- single worker debug on single environment

  ```
  python src/benchmarks/tartanair_og.py \
  --train_ann_file /datadrive/commondatasets/tartanair-release1/train_ann_debug_ratnesh_100_frames.json \
  --benchmark_results_file benchmark_results.csv \
  --modalities image_left depth_left flow_flow \
  --train_transform TartanAirVideoTransformWithAugmentation \
  --batch_size 1 \
  --num_workers 0 \
  --num_seq 1 \
  --seq_len 1
  ```

- pytorch bottleneck

  ```
  python -m torch.utils.bottleneck \
  src/benchmarks/tartanair_og.py \
  --train_ann_file /datadrive/commondatasets/tartanair-release1/train_ann_debug_ratnesh_100_frames.json \
  --benchmark_results_file benchmark_results.csv \
  --modalities image_left depth_left flow_flow \
  --train_transform TartanAirVideoTransformWithAugmentation \
  --batch_size 1 \
  --num_workers 0 \
  --num_seq 1 \
  --seq_len 1
  ```

- cProfile:
  uncomment `cprofile(args)` and comment `main(args)` in `src/benchmarks_dataloader.py`

- line_profiler

  - install

    ```
    conda install -c anaconda line_profiler
    ```

  - add @profile macro before function.
    For an example, uncomment @profile in `build.py -> class TartanAirVideoTransformWithAugmentation -> __call__` and `tartanair_video.py -> class TartanAirVideoDataset -> `__getitem__\`

  - run:

    - step 1

      ```
      kernprof -l \
          src/benchmarks/tartanair_og.py \
          --train_ann_file /datadrive/commondatasets/tartanair-release1/train_ann_debug_ratnesh_100_frames.json \
          --benchmark_results_file benchmark_results.csv \
          --modalities image_left \
          --batch_size 1 \
          --num_workers 0 \
          --num_seq 16 \
          --seq_len 1
      ```

    - step 2

      ```
      python -m line_profiler benchmark_tartanair_dataloader.py.lprof
      ```

      Example results:

      ```
      (dataloader-benchmark) azureuser@linux-nc24rsv3-0:/datadrive/projects/dataloader-benchmark$ python -m line_profiler benchmark_tartanair_dataloader.py.lprof
      Timer unit: 1e-06 s

      Total time: 3.40855 s
      File: /datadrive/projects/dataloader-benchmark/src/data/build.py
      Function: __call__ at line 407

      Line #      Hits         Time  Per Hit   % Time  Line Contents
      ==============================================================
        407                                               @profile
        408                                               def __call__(self, item):
        409                                                   # TODO: Need a better visualization of these augmentations.
        410                                                   # 1. Color jittering
        411       101        452.0      4.5      0.0          if self.do_color_jitter:
        412                                                       # Asymmetric.
        413       101        888.0      8.8      0.0              if np.random.rand() < self.asymmetric_color_aug_prob:
        414        27     721222.0  26711.9     21.2                  images = [np.array(self.color_jitter(x)) for x in item["image_left"]]
        415                                                       # Symmetric.
        416celse:
        417       148       5898.0     39.9      0.2                  image_stack = np.concatenate(
        418        74      37050.0    500.7      1.1                      [np.array(x) for x in item["image_left"]], axis=0
        419                                                           )  # Shape: [H,W,C]*D -> [D*H,W,C].
        420       148      35557.0    240.2      1.0                  image_stack = np.array(
        421        74    1944922.0  26282.7     57.1                      self.color_jitter(Image.fromarray(image_stack)), dtype=np.uint8
        422                                                           )
        423       148       5475.0     37.0      0.2                  images = np.split(
        424        74        254.0      3.4      0.0                      image_stack, len(item["image_left"]), axis=0
        425                                                           )  # Shape: [D*H,W,C] -> [H,W,C]*D.
        426                                                   else:
        427                                                       images = [np.array(x) for x in item["image_left"]]
        428
        429                                                   # 2. Flipping
        430       101        411.0      4.1      0.0          if "flow_flow" in self.modalities:
        431                                                       flows = item["flow_flow"]
        432       101        227.0      2.2      0.0          if "depth_left" in self.modalities:
        433                                                       depths = item["depth_left"]
        434       101        215.0      2.1      0.0          if "seg_left" in self.modalities:
        435                                                       segs = item["seg_left"]
        436
        437       101        227.0      2.2      0.0          if self.do_flip:
        438       101        785.0      7.8      0.0              if np.random.rand() < self.h_flip_prob:  # h-flip
        439        54        323.0      6.0      0.0                  images = [x[:, ::-1] for x in images]  # Shape: [H,W,C].
        440        54        120.0      2.2      0.0                  if "flow_flow" in self.modalities:
        441                                                               flows = [
        442                                                                   x[:, ::-1] * [-1.0, 1.0] for x in item["flow_flow"]
        443                                                               ]  # Shape: [H,W,2].
        444        54        113.0      2.1      0.0                  if "depth_left" in self.modalities:
        445                                                               depths = [x[:, ::-1] for x in item["depth_left"]]  # Shape: [H,W,1].
        446        54        138.0      2.6      0.0                  if "seg_left" in self.modalities:
        447                                                               segs = [x[:, ::-1] for x in item["seg_left"]]  # Shape: [H,W,1].
        448       101        300.0      3.0      0.0              if np.random.rand() < self.v_flip_prob:  # v-flip
        449        10         49.0      4.9      0.0                  images = [x[::-1, :] for x in images]
        450        10         24.0      2.4      0.0                  if "flow_flow" in self.modalities:
        451                                                               flows = [x[::-1, :] * [1.0, -1.0] for x in item["flow_flow"]]
        452        10         22.0      2.2      0.0                  if "depth_left" in self.modalities:
        453                                                               depths = [x[::-1, :] for x in item["depth_left"]]
        454        10         21.0      2.1      0.0                  if "seg_left" in self.modalities:
        455                                                               segs = [x[::-1, :] for x in item["seg_left"]]
        456
        457                                                   # 3. Standard transformations
        458       101     291773.0   2888.8      8.6          images = [Image.fromarray(x) for x in images]
        459
        460       101        315.0      3.1      0.0          transformed_item = {}
        461       202      11404.0     56.5      0.3          transformed_item["image_left"] = torch.stack(
        462       101     349196.0   3457.4     10.2              [self.image_transform(x) for x in images], dim=1
        463                                                   )  # Shape: [H,W,C]*D -> [C,H,W]*D -> [C,D,H,W].
        464       101        445.0      4.4      0.0          if "flow_flow" in self.modalities:
        465                                                       transformed_item["flow_flow"] = torch.stack(
        466                                                           [self.flow_transform(x) for x in flows], dim=1
        467                                                       )  # Shape: [H,W,2]*D -> [2,H,W]*D -> [2,D,H,W].
        468       101        252.0      2.5      0.0          if "depth_left" in self.modalities:
        469                                                       transformed_item["depth_left"] = torch.stack(
        470                                                           [self.depth_transform(x) for x in depths], dim=1
        471                                                       )  # Shape: [H,W,2]*D -> [2,H,W]*D -> [2,D,H,W].
        472       101        234.0      2.3      0.0          if "seg_left" in self.modalities:
        473                                                       transformed_item["seg_left"] = torch.stack(
        474                                                           [self.depth_transform(x) for x in segs], dim=1
        475                                                       )  # Shape: [H,W,2]*D -> [2,H,W]*D -> [2,D,H,W].
        476
        477       101        236.0      2.3      0.0          return transformed_item
      ```

## Benchmarking ffcv

### Installing:

```
sudo apt install libturbojpeg-dev libopencv-dev

# cupy has pre-compiled binaries for your cuda version.

# find your version and modify the command below accordingly

# docs: https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-pypi

pip install cupy-cuda116

# could be pip install cupy-cuda117

pip install ffcv numba opencv-python

```
