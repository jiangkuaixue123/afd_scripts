#!/bin/bash
# 运行 PyTorch new_group 示例（3 个 rank）
# 单机 3 GPU: torchrun --nproc_per_node=3 pytorch_new_group_demo.py
# 单机无 GPU (CPU): 使用 gloo 后端，同上

torchrun --nproc_per_node=3 pytorch_new_group_demo.py
