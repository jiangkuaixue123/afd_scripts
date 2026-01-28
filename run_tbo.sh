#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5
export VLLM_TORCH_PROFILER_DIR=./vllm_profile_afd
export VLLM_TORCH_PROFILER_WITH_STACK=0
export CUDA_LAUNCH_BLOCKING=1
vllm serve "/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite"  --data_parallel_size=2 --enable_expert_parallel \
       	--ubatch-size=3 --dbo-prefill-token-threshold 12 --dbo-decode-token-threshold 12 \
	--all2all-backend="deepep_low_latency"
#vllm serve "/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite"  --data_parallel_size=2 --enable_expert_parallel --enforce_eager --enable-dbo --num_of_microbatches=3 --dbo-prefill-token-threshold 12 --dbo-decode-token-threshold 12 --all2all-backend="deepep_low_latency"
