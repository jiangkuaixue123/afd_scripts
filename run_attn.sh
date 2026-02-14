#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
export VLLM_TORCH_PROFILER_DIR=./vllm_profile_afd
export VLLM_TORCH_PROFILER_WITH_STACK=0
# export CUDA_LAUNCH_BLOCKING=1
# export VLLM_LOGGING_LEVEL=DEBUG

export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=ALL
# 使用 %h (hostname) 和 %p (pid) 来区分不同进程的日志
# NCCL 原生支持 %h 和 %p，但不支持 %r (rank) 除非在特定环境下
export NCCL_DEBUG_FILE=nccl_rank_attn_%h_%p.log
export NCCL_DEBUG_TIMESTAMP=1
# 10.248.12.106
rm -rf /tmp/torchinductor_root/
rm -rf ~/.cache/vllm/
# -cc.cudagraph_mode=NONE \
# -cc.mode=0 \

# export NCCL_MAX_NCHANNEL=64
# export NCCL_BUFFSIZE=16777216
export BATCH_SIZE=${2:-64}
export VLLM_NCCL_SO_PATH=/home/fq9hpsac/fq9hpsacuser03/sources/nccl/build/lib/libnccl.so.2.29.3
export NCCL_GRAPH_MIXING_SUPPORT=1
# export NCCL_P2P_NET_CHUNKSIZE=262144
export NCCL_P2P_NET_CHUNKSIZE=524288
# export CUDA_LAUNCH_BLOCKING=1
export NCCL_NET_PLUGIN=none

vllm serve "/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite" \
    --max-num-batched-tokens $BATCH_SIZE \
    --data-parallel-size=2 \
    --enable_expert_parallel \
    --enable-dbo \
    --dbo-prefill-token-threshold 12 \
    --dbo-decode-token-threshold 2 \
    --port 8022 \
    --no-enable-prefix-caching \
    --compilation-config '{
		"cudagraph_mode": "FULL_DECODE_ONLY",
		"cudagraph_capture_sizes": ['$BATCH_SIZE']
	}' \
    --kv-transfer-config '{
        "kv_connector": "MooncakeConnector",
        "kv_role": "kv_consumer"
    }' \
    --afd-config '{
        "afd_connector":"p2pconnector",
        "afd_role": "attention",
        "afd_host":"10.248.12.142",
        "afd_port":"29521",
        "num_afd_stages":"2",
        "afd_extra_config":{
            "afd_size":"2A2F"
        }
    }' > attn.log 2>&1 &

# vllm serve "/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite" \
#     --data-parallel-size=2 \
#     --enable_expert_parallel \
#     --enable-dbo \
#     --dbo-prefill-token-threshold 12 \
#     --dbo-decode-token-threshold 2 \
#     --port 8022 \
#     --afd-config '{
#         "afd_connector":"p2pconnector",
#         "afd_role": "attention",
#         "afd_host":"127.0.0.1",
#         "afd_port":"29521",
#         "num_afd_stages":"2",
#         "afd_extra_config":{
#             "afd_size":"2A2F"
#         }
#     }' > attn.log 2>&1 &
