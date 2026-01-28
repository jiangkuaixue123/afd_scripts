#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
export VLLM_TORCH_PROFILER_DIR=./vllm_profile_afd
export VLLM_TORCH_PROFILER_WITH_STACK=0
# export VLLM_LOGGING_LEVEL=DEBUG
# export GLOO_DEBUG=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export GLOO_TIMEOUT_SECS=10
# export TORCH_CPP_LOG_LEVEL=INFO
# vllm serve "/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite" \
#      --data_parallel_size=4 \
#      --enable_expert_parallel \
#      --enforce_eager \
#      --enable-dbo --dbo-prefill-token-threshold 12 --dbo-decode-token-threshold 2 \
#      --afd-config '{"afd_connector":"p2pconnector", "afd_role": "attention", "afd_host":"127.0.0.1", "afd_port":"29510","num_afd_stages":"2","afd_extra_config":{"afd_size":"4A4F"}}' > attn.log 2>&1 &

vllm serve "/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite" \
    --data-parallel-size=2 \
    --enable_expert_parallel \
    --enforce_eager \
    --enable-dbo \
    --dbo-prefill-token-threshold 12 \
    --dbo-decode-token-threshold 2 \
    --afd-config '{
        "afd_connector":"p2pconnector",
        "afd_role": "attention",
        "afd_host":"10.248.12.106",
        "afd_port":"29510",
        "num_afd_stages":"2",
        "afd_extra_config":{
            "afd_size":"2A2F"
        }
    }' > attn.log 2>&1 &
