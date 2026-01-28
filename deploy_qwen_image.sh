#!/bin/bash
#

export VLLM_LOGGING_LEVEL=DEBUG
vllm serve Qwen3-Image-2512 --omni --port 8091 > qwen_images.log 2>&1 &
