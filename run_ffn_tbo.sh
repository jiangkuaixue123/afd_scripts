#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
vllm fserver "/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite" --data_parallel_size=2 \
	--enable_expert_parallel --enforce_eager --afd-config '{"afd_connector":"p2pconnector", "num_afd_stages":"3", "afd_role": "ffn", "afd_host":"127.0.0.1", "afd_port":"29507", "afd_extra_config":{"afd_size":"2A2F"}}' > ffn.log 2>&1 &
