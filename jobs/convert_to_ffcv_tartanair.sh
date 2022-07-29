dataset=tartanair;
tartanair_ann=/mnt/data/tartanair-release1/train_ann_debug_ratnesh_100_frames.json;
tartanair_output_beton_file=/mnt/results/tartanair-ffcv/train_ann_debug_ratnesh_100_frames.beton;
dataset=$dataset tartanair_ann=$tartanair_ann tartanair_output_beton_file=$tartanair_output_beton_file amlt run -y jobs/tartanair_convert_to_ffcv.yaml tartan_air_convert_to_ffcv;

dataset=tartanair;
tartanair_ann=/mnt/data/tartanair-release1/train_ann_debug_ratnesh.json;
tartanair_output_beton_file=/mnt/results/tartanair-ffcv/train_ann_debug_ratnesh.beton;
dataset=$dataset tartanair_ann=$tartanair_ann tartanair_output_beton_file=$tartanair_output_beton_file amlt run -y jobs/tartanair_convert_to_ffcv.yaml tartan_air_convert_to_ffcv;

dataset=tartanair
tartanair_ann=/mnt/data/tartanair-release1/train_ann_abandonedfactory.json
tartanair_output_beton_file=/mnt/results/tartanair-ffcv/train_ann_abandonedfactory.beton
dataset=$dataset tartanair_ann=$tartanair_ann tartanair_output_beton_file=$tartanair_output_beton_file amlt run -y jobs/tartanair_convert_to_ffcv.yaml tartan_air_convert_to_ffcv;

dataset=tartanair
tartanair_ann=/mnt/data/tartanair-release1/train_ann_abandonedfactory_easy.json
tartanair_output_beton_file=/mnt/results/tartanair-ffcv/train_ann_abandonedfactory_easy.beton
dataset=$dataset tartanair_ann=$tartanair_ann tartanair_output_beton_file=$tartanair_output_beton_file amlt run -y jobs/tartanair_convert_to_ffcv.yaml tartan_air_convert_to_ffcv;

dataset=tartanair;
tartanair_ann=/mnt/data/tartanair-release1/train_ann_abandonedfactory_hard.json;
tartanair_output_beton_file=/mnt/results/tartanair-ffcv/train_ann_abandonedfactory_hard.beton;
dataset=$dataset tartanair_ann=$tartanair_ann tartanair_output_beton_file=$tartanair_output_beton_file amlt run -y jobs/tartanair_convert_to_ffcv.yaml tartan_air_convert_to_ffcv;
