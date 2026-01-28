#!/bin/bash
#
python /home/fq9hpsac/fq9hpsacuser03/sources/vllm-omni/examples/online_serving/qwen3_omni/openai_chat_completion_client_for_multimodal_generation.py \
	    --query-type use_image \
	    --model Qwen3-Omni-30B-A3B-Instruct \
	    --image-path /home/fq9hpsac/fq9hpsacuser03/media.png \
	    --prompt "What are the main activities shown in this picture?"
