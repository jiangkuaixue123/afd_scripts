#!/bin/bash

# export GLOO_USE_SYNC_COMM=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export GLOO_TIMEOUT_SECS=10
# export VLLM_LOGGING_LEVEL=DEBUG
# export GLOO_DEBUG=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL


export CUDA_VISIBLE_DEVICES=$1
vllm serve "/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite" \
    -dp=2 \
    --enable_expert_parallel \
    --enforce_eager \
    --afd-config '{
        "afd_connector":"p2pconnector",
        "num_afd_stages":"2",
        "afd_role": "ffn",
        "afd_host":"10.248.12.106",
        "afd_port":"29510",
        "afd_extra_config":{
            "afd_size":"2A2F"
        }
    }' > ffn.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=4,5,6,7
# vllm serve "/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite" \
#     -dp=4 \
#     --enable_expert_parallel \
#     --enforce_eager \
#     --afd-config '{
#         "afd_connector":"p2pconnector",
#         "num_afd_stages":"2",
#         "afd_role": "ffn",
#         "afd_host":"127.0.0.1",
#         "afd_port":"29510",
#         "afd_extra_config":{
#             "afd_size":"4A4F"
#         }
#     }' > ffn.log 2>&1 &

