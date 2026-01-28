#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5
VLLM_TORCH_PROFILER_DIR=./vllm_profile VLLM_TORCH_PROFILER_WITH_STACK=0 vllm serve --model="/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite" --data-parallel-size 2 --enable-expert-parallel
