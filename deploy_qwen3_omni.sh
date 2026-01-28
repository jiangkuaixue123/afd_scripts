#!/bin/bash
#

export VLLM_LOGGING_LEVEL=DEBUG
vllm serve Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 > omni.log 2>&1 &
