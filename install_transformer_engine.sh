#!/bin/bash

# Transformer Engine 简单安装脚本
# 解决cuDNN头文件缺失问题

echo "安装 transformer_engine[pytorch]..."

# 设置cuDNN头文件路径
export CPLUS_INCLUDE_PATH=/home/qianwenhao/.conda/envs/llamafactory/lib/python3.11/site-packages/nvidia/cudnn/include

# 激活环境
conda activate llamafactory

# 安装
pip install transformer_engine[pytorch]==2.1.0 --no-cache-dir

echo "安装完成！"
