#!/bin/bash


export LOCAL_RANK=$1
export VLLM_NCCL_SO_PATH=/home/fq9hpsac/fq9hpsacuser03/sources/nccl/build/lib/libnccl.so.2.29.3
export WORLD_SIZE=2
export RANK=1
export MASTER_ADDR=10.248.12.142
export MASTER_PORT=29500

python pynccl_demo.py > temp_attn.log 2>&1 &