#!/bin/bash
    # --enforce-eager \
export CUDA_VISIBLE_DEVICES=$1
export VLLM_TORCH_PROFILER_DIR=./vllm_profile
export VLLM_TORCH_PROFILER_WITH_STACK=0

export TORCH_LOGS="+dynamo"

# vllm serve --model="/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite" \
#     --data-parallel-size 2 \
#     -tp 1 \
#     --enable-expert-parallel \
# 	--max-num-batched-tokens 72 \
#     --compilation-config '{
# 		"cudagraph_mode": "FULL_DECODE_ONLY",
# 		"cudagraph_capture_sizes": [72]
# 	}' \
#     > normal.log 2>&1 &

vllm serve --model="/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite" \
    --data-parallel-size 2 \
    -tp 1 \
    --enable-expert-parallel \
	--max-num-batched-tokens 72 \
    --compilation-config '{
		"cudagraph_mode": "FULL_DECODE_ONLY",
		"cudagraph_capture_sizes": [72]
	}' \
    --kv-transfer-config '{
        "kv_connector": "MooncakeConnector",
        "kv_role": "kv_consumer"
    }' \
    > normal.log 2>&1 &