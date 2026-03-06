#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_DEBUG_FILE=nccl_rank_ffn_%h_%p.log
# export NCCL_DEBUG_TIMESTAMP=1
    # --enforce_eager \
# export TORCH_LOGS="+dynamo"
#export NCCL_MAX_NCHANNEL=64
#export NCCL_BUFFSIZE=16777216
# export VLLM_NCCL_SO_PATH=/home/fq9hpsac/fq9hpsacuser03/sources/nccl/build/lib/libnccl.so.2.29.3
export NCCL_GRAPH_MIXING_SUPPORT=1
# export NCCL_P2P_NET_CHUNKSIZE=262144
# export NCCL_P2P_NET_CHUNKSIZE=524288
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1

vllm serve "/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite" \
    -dp=2 \
    --enable_expert_parallel \
    --compilation-config '{
		"cudagraph_mode": "FULL_DECODE_ONLY",
		"cudagraph_capture_sizes": [256]
	}' \
    --enable-dbo \
    --dbo-prefill-token-threshold 12 \
    --dbo-decode-token-threshold 2 \
    --port 8021 \
    --afd-config '{
        "afd_connector":"p2pconnector",
        "num_afd_stages":"2",
        "afd_role": "ffn",
        "afd_host":"10.248.12.80",
        "afd_port":"29531",
        "afd_extra_config":{
            "afd_size":"4A2F"
        }
    }' > ffn.log 2>&1 &


# vllm serve "/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite" \
#     -dp=2 \
#     --port 8021 \
#     --enforce_eager \
#     --enable_expert_parallel \
#     --enable-dbo \
#     --dbo-prefill-token-threshold 12 \
#     --dbo-decode-token-threshold 2 \
#     --afd-config '{
#         "afd_connector":"p2pconnector",
#         "num_afd_stages":"2",
#         "afd_role": "ffn",
#         "afd_host":"127.0.0.1",
#         "afd_port":"29521",
#         "afd_extra_config":{
#             "afd_size":"2A2F"
#         }
#     }' > ffn.log 2>&1 &
