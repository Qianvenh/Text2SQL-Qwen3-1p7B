workdir=$(cd $(dirname $0); pwd)
configdir="$workdir/config/train"
scriptdir="$workdir/scripts"
WORLD_SIZE=4
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export DISABLE_VERSION_CHECK=1

log_name="qwen3_1p7B_$(date +'%Y%m%d_%H%M%S')_megatron.log"

nohup torchrun --nproc_per_node=$WORLD_SIZE $scriptdir/megatron_run_train.py \
	$configdir/qwen3_1p7B_deepspeed_megatron_lora_qv.yaml \
	> ./logs/$log_name 2>&1 &

echo "Training started, logging to logs/$log_name"
echo "PID: $!"
