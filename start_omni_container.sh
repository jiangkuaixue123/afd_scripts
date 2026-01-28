#!/bin/bash

docker run -itd --shm-size=64g --privileged=true \
--gpus all \
--name jcz-omni \
--net=host \
-w /home/fq9hpsac/fq9hpsacuser03 \
-v ~/.cache/huggingface:/root/.cache/huggingface \
-v ~/.cache/jcz:/root/.cache/jcz \
-v /home/fq9hpsac/fq9hpsacuser03:/home/fq9hpsac/fq9hpsacuser03 \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
nvcr.io/nvidia/pytorch:25.01-py3
