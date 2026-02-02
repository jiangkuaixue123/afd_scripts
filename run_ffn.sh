#!/bin/bash

# export GLOO_USE_SYNC_COMM=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export GLOO_TIMEOUT_SECS=10
# export VLLM_LOGGING_LEVEL=DEBUG
# export GLOO_DEBUG=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
    # -dp=2 \


export CUDA_VISIBLE_DEVICES=$1
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_DEBUG_FILE=nccl_rank_ffn_%h_%p.log
# export NCCL_DEBUG_TIMESTAMP=1

vllm serve "/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite" \
    -dp=2 \
    --enable_expert_parallel \
    --enforce_eager \
    --enable-dbo \
    --port 8021 \
    --dbo-prefill-token-threshold 12 \
    --dbo-decode-token-threshold 2 \
    --afd-config '{
        "afd_connector":"p2pconnector",
        "num_afd_stages":"2",
        "afd_role": "ffn",
        "afd_host":"127.0.0.1",
        "afd_port":"29521",
        "afd_extra_config":{
            "afd_size":"2A2F"
        }
    }' > ffn.log 2>&1 &

# vllm serve "/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite" \
#     --enable_expert_parallel \
#     --enforce_eager \
#     --enable-dbo \
#     --dbo-prefill-token-threshold 12 \
#     --dbo-decode-token-threshold 2 \
#     --afd-config '{
#         "afd_connector":"p2pconnector",
#         "num_afd_stages":"1",
#         "afd_role": "ffn",
#         "afd_host":"127.0.0.1",
#         "afd_port":"29510",
#         "afd_extra_config":{
#             "afd_size":"1A1F"
#         }
#     }' > ffn.log 2>&1 &