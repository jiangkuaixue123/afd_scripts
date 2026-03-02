#!/bin/bash
    # --enforce-eager \
export CUDA_VISIBLE_DEVICES=$1
export VLLM_TORCH_PROFILER_DIR=./vllm_profile
export VLLM_TORCH_PROFILER_WITH_STACK=0


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
rm -rf /tmp/torchinductor_root/


vllm serve --model="/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite" \
    --data-parallel-size 2 \
    -tp 1 \
    --enable-expert-parallel \
	--max-num-batched-tokens 256 \
    --no-enable-prefix-caching \
    --compilation-config '{
		"cudagraph_mode": "FULL_DECODE_ONLY",
		"cudagraph_capture_sizes": [256]
	}' \
    --kv-transfer-config '{
        "kv_connector": "MooncakeConnector",
        "kv_role": "kv_consumer"
    }' \
    > normal.log 2>&1 &

# --capture-range=cudaProfilerApi \
#     --capture-range-end repeat \

# nsys profile \
#     --trace-fork-before-exec=true \
#     --cuda-graph-trace=node \
#     -o ./normal_profile.nsys \
#     --delay 5 --duration 120 \
#     vllm serve --model="/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite" \
#     --data-parallel-size 2 \
#     -tp 1 \
#     --profiler-config.profiler cuda \
#     --enable-expert-parallel \
# 	--max-num-batched-tokens 256 \
#     --no-enable-prefix-caching \
#     --compilation-config '{
# 		"cudagraph_mode": "FULL_DECODE_ONLY",
# 		"cudagraph_capture_sizes": [256]
# 	}' \
#     --kv-transfer-config '{
#         "kv_connector": "MooncakeConnector",
#         "kv_role": "kv_consumer"
#     }' \
#     > normal.log 2>&1 &