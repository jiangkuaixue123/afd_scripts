#!/bin/bash
#
# vllm bench serve \
#   --backend vllm \
#   --model /home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite \
#   --endpoint /v1/completions \
#   --dataset-name burstgpt \
#   --dataset-path /root/.cache/jcz/BurstGPT_without_fails_2.csv \
#   --max-concurrency 60 \
#   --num-prompts 1000

vllm bench serve \
 --backend vllm \
 --model /home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite \
 --endpoint /v1/completions \
 --dataset-name random \
 --random-input-len 2 \
 --random-output-len 100 \
 --max-concurrency 512 \
 --num-prompts 10000
