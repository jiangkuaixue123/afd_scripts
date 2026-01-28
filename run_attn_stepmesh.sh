#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
export VLLM_TORCH_PROFILER_DIR=./vllm_profile_afd
export VLLM_TORCH_PROFILER_WITH_STACK=0


# -dp 2 \
# --enable-dbo \
# --dbo-prefill-token-threshold 12 \
# --dbo-decode-token-threshold 2 \
# --enforce-eager \

vllm serve /home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite \
    --enable_expert_parallel \
    --max-num-batched-tokens 72 \
	--enable-dbo \
	--dbo-prefill-token-threshold 12 \
	--dbo-decode-token-threshold 2 \
    --port 8012 \
	--compilation-config '{
		"cudagraph_mode": "FULL_DECODE_ONLY",
		"cudagraph_capture_sizes": [72]
	}' \
    --afd-config '{
        "afd_connector":"stepmeshconnector", 
        "afd_role": "attention", 
        "afd_host":"10.248.12.106", 
        "afd_port":"29512", 
        "num_ffn_servers": "1", 
        "num_attention_servers": "1", 
        "num_afd_stages":"1"
    }' > attn.log 2>&1 &
