#!/bin/bash

conda activate llamafactory
source activate llamafactory

path_to_megatron_model=saves/qwen3-1.7B-tp_dp_p2train/checkpoint-6312
path_to_output_hf_model=saves/qwen3-1.7B-tp_dp_p2train/lora/sft/hf_model

python src/convert_mg2hf.py \
	--checkpoint_path $path_to_megatron_model \
	--output_path $path_to_output_hf_model
