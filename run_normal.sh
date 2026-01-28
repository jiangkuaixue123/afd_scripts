#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
VLLM_TORCH_PROFILER_DIR=./vllm_profile VLLM_TORCH_PROFILER_WITH_STACK=0 vllm serve --model="/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite" \
	--data-parallel-size 2 \
	-tp 1 --enable-expert-parallel --enforce-eager
