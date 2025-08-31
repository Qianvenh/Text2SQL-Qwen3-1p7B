#!/bin/bash

source activate llamafactory

log_name="qwen3_1p7B_$(date +'%Y%m%d_%H%M%S').log"

FORCE_TORCHRUN=1 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
nohup llamafactory-cli train config/train/qwen3_1p7B_deepspeed_lora_qv.yaml \
	> logs/$log_name 2>&1 & 
echo "Training started, logging to logs/$log_name"
echo "PID: $!"
