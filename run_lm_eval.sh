#!/bin/bash
#

lm_eval run \
  --model local-completions \
  --model_args model=/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite,base_url=http://127.0.0.1:8022/v1/completions,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k \
  --limit 0.5 \
  --batch_size 128 \
  --log_samples \
  --output_path ./lm_eval_result


