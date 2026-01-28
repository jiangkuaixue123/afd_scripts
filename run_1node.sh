#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
vllm serve --model="/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite" \
	--data-parallel-size 4 \
    	--data-parallel-size-local 2 \
    	--data-parallel-address 127.0.0.1 \
    	--data-parallel-rpc-port 13345 \
	--enable-expert-parallel \
	--enforce-eager > node1.log 2>&1 &

