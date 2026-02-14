#!/usr/bin/env python3
"""
PyTorch NVSHMEM 双 rank 示例：使用 Triton 内核演示 put/get。
参照 PyTorch test/distributed/test_nvshmem_triton.py 的写法。

运行方式（需要 2 张 GPU 且支持 NVSHMEM）:
  torchrun --nproc_per_node=2 nvshmem_put_get_example.py

依赖: PyTorch 编译时启用 NVSHMEM，且环境已安装 NVSHMEM。
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

# Triton 与 NVSHMEM Triton 扩展（内核内需用 nvshmem.put / nvshmem.get）
import torch.distributed._symmetric_memory._nvshmem_triton as nvshmem
from torch._inductor.runtime.triton_compat import triton
from torch.distributed._symmetric_memory._nvshmem_triton import requires_nvshmem


def is_nvshmem_available():
    try:
        return symm_mem.is_nvshmem_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Triton JIT 内核：与 test_nvshmem_triton.py 一致
# ---------------------------------------------------------------------------

@requires_nvshmem
@triton.jit
def my_put_kernel(
    dest,
    src,
    nelems,
    pe,
):
    nvshmem.put(dest, src, nelems, pe)


@requires_nvshmem
@triton.jit
def my_get_kernel(
    dest,
    src,
    nelems,
    pe,
):
    nvshmem.get(dest, src, nelems, pe)


def main():
    # 使用 torchrun 时由环境变量提供
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 2))

    if world_size != 2:
        raise RuntimeError(
            "本示例仅支持 2 个 rank，请使用: torchrun --nproc_per_node=2 nvshmem_put_get_example.py"
        )

    if not is_nvshmem_available():
        print("NVSHMEM 不可用，跳过示例", file=sys.stderr)
        sys.exit(0)

    # 初始化进程组
    dist.init_process_group(backend="nccl")
    assert dist.get_world_size() == 2

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    # 设置 NVSHMEM 为对称内存后端（与 test 中 _init_device 一致）
    symm_mem.set_backend("NVSHMEM")

    group_name = dist.distributed_c10d._get_default_group().group_name
    symm_mem.enable_symm_mem_for_group(group_name)

    torch.manual_seed(42 + rank)
    peer = 1 - rank

    # -------------------------------------------------------------------------
    # 阶段 1: Put — Rank 0 put 数据到 Rank 1（对应 test_triton_put）
    # -------------------------------------------------------------------------
    nelems_put = 5
    dtype_put = torch.int64
    val_put = 42 + rank

    src_put = symm_mem.empty(nelems_put, dtype=dtype_put, device=device)
    dst_put = symm_mem.empty(nelems_put, dtype=dtype_put, device=device).fill_(-999)

    for i in range(nelems_put):
        src_put[i] = val_put * 10 + i

    symm_mem.rendezvous(src_put, group=group_name)
    symm_mem.rendezvous(dst_put, group=group_name)
    dist.barrier()

    if rank == 0:
        my_put_kernel[(1,)](
            dst_put,
            src_put,
            nelems_put,
            peer,
        )

    dist.barrier()

    if rank == 1:
        expected_put = [420 + i for i in range(nelems_put)]
        expected_tensor = torch.tensor(expected_put, device=device, dtype=dtype_put)
        if torch.equal(dst_put, expected_tensor):
            print(f"[Rank 1] PUT 接收成功: 收到来自 Rank 0 的数据 {expected_put}")
        else:
            print(f"[Rank 1] PUT 接收异常: 期望 {expected_put}, 得到 {dst_put.tolist()}")

    dist.barrier()

    # -------------------------------------------------------------------------
    # 阶段 2: Get — Rank 1 从 Rank 0 get 数据（对应 test_triton_get）
    # -------------------------------------------------------------------------
    numel_get = 8
    dtype_get = torch.int8
    val_get = 7

    inp_get = symm_mem.empty(numel_get, dtype=dtype_get, device=device).fill_(
        val_get if rank == 0 else -1
    )
    out_get = symm_mem.empty(numel_get, dtype=dtype_get, device=device).fill_(-1)
    symm_mem.rendezvous(inp_get, group=group_name)
    symm_mem.rendezvous(out_get, group=group_name)
    dist.barrier()

    if rank == 1:
        my_get_kernel[(1,)](
            out_get,
            inp_get,
            numel_get,
            peer,
        )

    dist.barrier()

    if rank == 1:
        expected_get = val_get * torch.ones(numel_get, dtype=dtype_get, device=device)
        if torch.equal(out_get, expected_get):
            print(f"[Rank 1] GET 成功: 已从 Rank 0 get 数据，校验一致 (value={val_get})")
        else:
            print(f"[Rank 1] GET 异常: 期望 {expected_get.tolist()}, 得到 {out_get.tolist()}")

    dist.barrier()
    print(f"[Rank {rank}] 示例结束")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
