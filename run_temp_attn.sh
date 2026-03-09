#!/bin/bash


export LOCAL_RANK=$1
# export VLLM_NCCL_SO_PATH=/home/fq9hpsac/fq9hpsacuser03/sources/nccl/build/lib/libnccl.so.2.29.3
export WORLD_SIZE=2
export RANK=1
export MASTER_ADDR=10.248.12.142
export MASTER_PORT=29500
# export NCCL_NET_PLUGIN=none
# export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=ALL
# # 使用 %h (hostname) 和 %p (pid) 来区分不同进程的日志
# # NCCL 原生支持 %h 和 %p，但不支持 %r (rank) 除非在特定环境下
# export NCCL_DEBUG_FILE=nccl_rank_attn_%h_%p.log
python pynccl_p2p_cudagraph.py > temp_attn.log 2>&1 &