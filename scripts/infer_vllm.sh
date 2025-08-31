#!/bin/bash

conda activate llamafactory
source activate llamafactory

python src/vllm_infer.py \
	./config/vllm_infer/vllm_infer_config.yaml