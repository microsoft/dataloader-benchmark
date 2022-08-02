num_cpus=32
for use in forecast; do
  for split in train val test ; do
    npz_path=/mnt/data/1.40625deg_yearly_np/$split
    npy_path=/mnt/data_out/1.40625deg_yearly_np/$use/$split
    echo -e Next Job
    echo -e use: $use '\n'split: $split '\n'npz_path: $npz_path '\n'npy_path: $npy_path
    num_cpus=$num_cpus use=$use split=$split npz_path=$npz_path npy_path=$npy_path amlt run -y climate_npz_to_npy.yaml climate_npz_to_npy -t F32s-v2-ded;
    done
done

num_cpus=72
for use in pretrain ; do
  for split in train val test ; do
    npz_path=/mnt/data/1.40625deg_yearly_np/$split
    npy_path=/mnt/data_out/1.40625deg_yearly_np/$use/$split
    echo -e Next Job
    echo -e use: $use '\n'split: $split '\n'npz_path: $npz_path '\n'npy_path: $npy_path
    num_cpus=$num_cpus use=$use split=$split npz_path=$npz_path npy_path=$npy_path amlt run -y climate_npz_to_npy.yaml climate_npz_to_npy -t F72s-v2-lp;
    done
done
