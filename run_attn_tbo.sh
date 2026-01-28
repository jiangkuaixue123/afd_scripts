#!/bin/bash


export CUDA_VISIBLE_DEVICES=$1
export VLLM_TORCH_PROFILER_DIR=./vllm_profile_afd
export VLLM_TORCH_PROFILER_WITH_STACK=0
export CUDA_LAUNCH_BLOCKING=1
export VLLM_ATTENTION_BACKEND="TRITON_MLA"
vllm serve "/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite" --gpu-memory-utilization=0.75  --data_parallel_size=2 --enable_expert_parallel --enforce_eager --enable-dbo --num-of-microbatches=3  --dbo-prefill-token-threshold 12 --dbo-decode-token-threshold 6 --afd-config '{"afd_connector":"p2pconnector", "afd_role": "attention", "afd_host":"127.0.0.1", "afd_port":"29507","num_afd_stages":"3","afd_extra_config":{"afd_size":"2A2F"}}' > attn.log 2>&1 &
