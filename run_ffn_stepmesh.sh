#!/bin/bash

# export GLOO_USE_SYNC_COMM=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export GLOO_TIMEOUT_SECS=10
# export VLLM_LOGGING_LEVEL=DEBUG
# export GLOO_DEBUG=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# -tp 2 \
# --enforce_eager \
# --compilation-config '{
# 		"cudagraph_mode": "FULL_DECODE_ONLY",
# 		"cudagraph_capture_sizes": [72]
# 	}' \

export CUDA_VISIBLE_DEVICES=$1
vllm fserver /home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite \
    --enable_expert_parallel \
    --max-num-batched-tokens 72 \
	--enable-dbo \
	--dbo-prefill-token-threshold 12 \
	--dbo-decode-token-threshold 2 \
	--compilation-config '{
		"cudagraph_mode": "FULL_DECODE_ONLY",
		"cudagraph_capture_sizes": [72]
	}' \
    --afd-config '{
        "afd_connector":"stepmeshconnector",
        "afd_role": "ffn",
        "afd_host":"10.248.12.106",
        "afd_port":"29512",
        "num_ffn_servers": "1",
        "num_attention_servers": "1",
        "num_afd_stages":"1"
    }' \
    > ffn.log 2>&1 &
