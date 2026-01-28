#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_TORCH_PROFILER_DIR=./vllm_profile_afd
export VLLM_TORCH_PROFILER_WITH_STACK=0
export TORCH_LOGS="graph_breaks"
export VLLM_LOGGING_LEVEL=DEBUG
export CUDA_LAUNCH_BLOCKING=1
TORCH_LOGS="graph_breaks" vllm serve "/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite"  --data_parallel_size=2 --enable_expert_parallel \
       	--enable-dbo --dbo-prefill-token-threshold 12 --dbo-decode-token-threshold 12 \
	--all2all-backend="deepep_low_latency"
	#--all2all-backend="deepep_high_throughput"
#vllm serve "/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite"  --data_parallel_size=2 --enable_expert_parallel --enforce_eager --enable-dbo --num_of_microbatches=3 --dbo-prefill-token-threshold 12 --dbo-decode-token-threshold 12 --all2all-backend="deepep_low_latency"
