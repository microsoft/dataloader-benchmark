for use in pretrain forecast; do
  for split in train val test ; do
    npz_path=/mnt/data/1.40625deg_yearly_np/$split
    npy_path=/mnt/data_out/1.40625deg_yearly_np/$use/$split
    echo -e Next Job
    echo -e use: $use '\n'split: $split '\n'npz_path: $npz_path '\n'npy_path: $npy_path
    use=$use split=$split npz_path=$npz_path npy_path=$npy_path amlt run -y climate_npz_to_npy.yaml climate_npz_to_npy;
    done
done
