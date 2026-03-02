#!/usr/bin/env python3
"""
PyTorch 分布式示例：3 个 rank，使用 new_group 分别创建 NCCL 和 Gloo 后端的进程组
运行方式: torchrun --nproc_per_node=3 pytorch_new_group_demo.py
"""

import os
import torch
import torch.distributed as dist


def main():
    # 根据环境变量初始化分布式（由 torchrun 自动设置）
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 默认使用 NCCL（GPU），无 GPU 时使用 Gloo（CPU）
    if torch.cuda.is_available():
        backend = "nccl"
        device = torch.device(f"cuda:{local_rank}")
    else:
        backend = "gloo"
        device = torch.device("cpu")

    dist.init_process_group(backend=backend)
    torch.cuda.set_device(local_rank) if torch.cuda.is_available() else None

    print(f"[Rank {rank}] 初始化完成, 默认后端: {backend}, world_size: {world_size}")

    # ========== 创建 NCCL 后端的 new_group (rank 0, 1, 2 全体) ==========
    # 注意: new_group 是集体操作，所有进程都必须调用
    nccl_ranks = [0, 1, 2]
    if torch.cuda.is_available():
        nccl_group = dist.new_group(ranks=nccl_ranks, backend="nccl")
        if rank in nccl_ranks:
            print(f"[Rank {rank}] 创建 NCCL 进程组成功")
    else:
        nccl_group = None
        print(f"[Rank {rank}] 跳过 NCCL 组 (无 CUDA)")

    # ========== 创建 Gloo 后端的 new_group (rank 0, 1, 2 全体) ==========
    gloo_ranks = [0, 1, 2]
    gloo_group = dist.new_group(ranks=gloo_ranks, backend="gloo")
    if rank in gloo_ranks:
        print(f"[Rank {rank}] 创建 Gloo 进程组成功 (子组 rank: {gloo_ranks.index(rank)})")
    else:
        print(f"[Rank {rank}] 不在 Gloo 子组中")

    # ========== 在 Gloo 组内做一次 all_reduce 演示 ==========
    if rank in gloo_ranks:
        tensor = torch.tensor([rank + 1.0], dtype=torch.float32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=gloo_group)
        print(f"[Rank {rank}] Gloo 组 all_reduce 结果: {tensor.item()} (应为 1+2+3=6)")

    # ========== 在 NCCL 组内做一次 all_reduce 演示（若有 GPU）==========
    if nccl_group is not None and torch.cuda.is_available():
        tensor = torch.tensor([rank + 1.0], dtype=torch.float32, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=nccl_group)
        print(f"[Rank {rank}] NCCL 组 all_reduce 结果: {tensor.item()} (应为 1+2+3=6)")

    dist.barrier()
    dist.destroy_process_group()
    print(f"[Rank {rank}] 完成并退出")


if __name__ == "__main__":
    main()
