#!/usr/bin/env python3
"""
Example of NVSHMEM with CUDA Graph support.

Key pattern:
- Sender (Rank 1): Uses CUDA graph to capture put(data) + put(signal)
- Receiver (Rank 0): Uses GPU spin-wait kernel (can also be captured in graph if needed)

This demonstrates a fully GPU-resident communication pattern suitable for CUDA graphs.

Requirements:
  - H100+ GPU
  - NVSHMEM library (pip install nvidia-nvshmem-cu12)
  - RDMA network (InfiniBand or RoCE)

Run (2 GPU on one machine):
  torchrun --nproc_per_node=2 symmetric_memory_nvshmem_cudagraph.py

Run across 2 node:
  MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 RANK=0 WORLD_SIZE=2 python symmetric_memory_nvshmem_cudagraph.py
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

if not symm_mem.is_nvshmem_available():
    print("ERROR: NVSHMEM not available. Please install nvidia-nvshmem-cu12")
    sys.exit(1)

import triton
import triton.language as tl

import torch.distributed._symmetric_memory._nvshmem_triton as nvshmem
from torch.distributed._symmetric_memory._nvshmem_triton import requires_nvshmem


# =============================================================================
# Triton Kernels
# =============================================================================


@requires_nvshmem
@triton.jit
def put_data_and_signal_kernel(
    data_ptr,  # Destination data buffer (on remote peer)
    signal_ptr,  # Destination signal buffer (on remote peer)
    src_data_ptr,  # Source data buffer (local)
    src_signal_val,  # Signal value to send
    nelems,  # Number of elements in data
    peer,  # Target peer rank
):
    """
    Put data and then put signal value atomically.
    Both operations are captured in CUDA graph.
    """
    # Put data
    nvshmem.put(data_ptr, src_data_ptr, nelems, peer)
    # Put signal value (1 element, int64)
    nvshmem.put(signal_ptr, src_signal_val, 1, peer)


@requires_nvshmem
@triton.jit
def spin_wait_kernel(
    signal_ptr,  # Signal buffer to wait on
    expected_val,  # Expected signal value
):
    """
    GPU-side spin-wait: polls signal until it reaches expected value.
    This kernel blocks until the signal arrives.

    IMPORTANT: This can be captured in CUDA graph because:
    - It's just a GPU kernel with memory loads
    - No CPU synchronization involved
    - No external NVSHMEM wait APIs
    """
    signal_addr = signal_ptr.to(tl.pointer_type(tl.int64))

    # Spin-wait loop
    while True:
        current_val = tl.load(signal_addr)
        if current_val == expected_val:
            break
        # 可选：添加 nanosleep 减少 bus traffic
        tl.nanosleep(100)  # 100ns


@requires_nvshmem
@triton.jit
def process_data_kernel(
    data_ptr,  # Data buffer
    output_ptr,  # Output buffer
    nelems,  # Number of elements
    scale_factor,  # Scale factor for processing
):
    """
    Simple data processing kernel: multiply by scale_factor.
    """
    pid = tl.program_id(0)

    # Process data in blocks
    block_size = 1024
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < nelems

    # Load, process, store
    data = tl.load(data_ptr + offsets, mask=mask)
    result = data * scale_factor
    tl.store(output_ptr + offsets, result, mask=mask)


# =============================================================================
# Main
# =============================================================================


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2, "This example expects 1 GPU per rank"

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)

    print(f"[Rank {rank}] Initializing NVSHMEM backend...")
    symm_mem.set_backend("NVSHMEM")

    group_name = dist.group.WORLD.group_name
    symm_mem.enable_symm_mem_for_group(group_name)

    # Configuration
    nelems = 64 * 2048  # 131072 elements
    dtype = torch.bfloat16
    signal_dtype = torch.int64

    print(f"[Rank {rank}] Creating symmetric memory buffers...")

    # Create symmetric memory buffers
    data_buffer = symm_mem.empty(nelems, dtype=dtype, device=device)
    signal_buffer = symm_mem.empty(1, dtype=signal_dtype, device=device)

    # Rendezvous
    symm_mem.rendezvous(data_buffer, group=dist.group.WORLD)
    symm_mem.rendezvous(signal_buffer, group=dist.group.WORLD)

    print(f"[Rank {rank}] Symmetric memory rendezvous completed")

    # =========================================================================
    # Phase 1: Warmup (ensure kernels are compiled)
    # =========================================================================
    print(f"\n[Rank {rank}] === Phase 1: Warmup ===")

    # Initialize data
    torch.manual_seed(42 + rank)
    if rank == 1:
        data_buffer.copy_(torch.randn(nelems, dtype=dtype, device=device))
    signal_buffer.zero_()

    dist.barrier()

    # Warmup: run kernels once to compile them
    signal_val_tensor = torch.tensor([12345], dtype=signal_dtype, device=device)

    if rank == 1:
        print(f"[Rank {rank}] Warmup: put_data_and_signal_kernel...")
        put_data_and_signal_kernel[(1,)](
            data_buffer,
            signal_buffer,
            data_buffer,
            signal_val_tensor,
            nelems,
            0,  # peer: rank 0
        )

    if rank == 0:
        print(f"[Rank {rank}] Warmup: spin_wait_kernel...")
        spin_wait_kernel[(1,)](
            signal_buffer,
            12345,
        )
        print(f"[Rank {rank}] Warmup: signal received!")

    torch.cuda.synchronize()
    dist.barrier()
    print(f"[Rank {rank}] Warmup completed")

    # =========================================================================
    # Phase 2: Capture CUDA Graph (Sender only)
    # =========================================================================
    if rank == 1:
        print(f"\n[Rank {rank}] === Phase 2: Capture CUDA Graph ===")

        # Reset for graph capture
        torch.manual_seed(100 + rank)
        data_buffer.copy_(torch.randn(nelems, dtype=dtype, device=device))
        signal_buffer.zero_()

        # Capture graph
        sender_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(sender_graph):
            put_data_and_signal_kernel[(1,)](
                data_buffer,
                signal_buffer,
                data_buffer,
                signal_val_tensor,
                nelems,
                0,
            )
        print(f"[Rank {rank}] CUDA Graph captured!")

    # =========================================================================
    # Phase 3: Capture CUDA Graph (Receiver)
    # =========================================================================
    if rank == 0:
        print(f"\n[Rank {rank}] === Phase 3: Capture Receiver Graph ===")

        # Reset for graph capture
        signal_buffer.zero_()

        # Capture receiver graph (spin_wait + process)
        receiver_graph = torch.cuda.CUDAGraph()
        output_buffer = torch.empty_like(data_buffer)

        with torch.cuda.graph(receiver_graph):
            # Wait for signal
            spin_wait_kernel[(1,)](
                signal_buffer,
                12345,
            )
            # Process data (example: multiply by 2.0)
            process_data_kernel[(128,)](
                data_buffer,
                output_buffer,
                nelems,
                2.0,
            )
        print(f"[Rank {rank}] Receiver Graph captured!")

    dist.barrier()

    # =========================================================================
    # Phase 4: Replay Graphs
    # =========================================================================
    print(f"\n[Rank {rank}] === Phase 4: Replay Graphs ===")

    num_iterations = 5

    for i in range(num_iterations):
        print(f"\n[Rank {rank}] --- Iteration {i} ---")

        # Reset signal
        if rank == 0:
            signal_buffer.zero_()
            output_buffer.zero_()

        # Update data on sender
        if rank == 1:
            torch.manual_seed(200 + i)
            data_buffer.copy_(torch.randn(nelems, dtype=dtype, device=device))

        dist.barrier()

        # Replay graphs
        if rank == 1:
            print(f"[Rank {rank}] Replaying sender graph...")
            sender_graph.replay()
            print(f"[Rank {rank}] Sender graph replay completed")

        if rank == 0:
            print(f"[Rank {rank}] Replaying receiver graph...")
            receiver_graph.replay()
            print(f"[Rank {rank}] Receiver graph replay completed")

            # Verify result
            torch.manual_seed(200 + i)
            expected_data = torch.randn(nelems, dtype=dtype, device=device) * 2.0
            if torch.allclose(output_buffer, expected_data, rtol=1e-2, atol=1e-2):
                print(f"[Rank {rank}] Iteration {i} PASSED!")
            else:
                max_diff = (output_buffer - expected_data).abs().max().item()
                print(f"[Rank {rank}] Iteration {i} FAILED: max_diff={max_diff}")

        dist.barrier()

    print(f"\n[Rank {rank}] All iterations completed!")

    # =========================================================================
    # Phase 5: Latency Measurement
    # =========================================================================
    print(f"\n[Rank {rank}] === Phase 5: Latency Measurement ===")

    # Warmup for timing
    for _ in range(3):
        if rank == 0:
            signal_buffer.zero_()
        dist.barrier()
        if rank == 1:
            sender_graph.replay()
        if rank == 0:
            receiver_graph.replay()
        dist.barrier()

    # Timing
    import time

    num_timed_iterations = 100
    start = time.perf_counter()

    for _ in range(num_timed_iterations):
        if rank == 0:
            signal_buffer.zero_()
        dist.barrier()
        if rank == 1:
            sender_graph.replay()
        if rank == 0:
            receiver_graph.replay()
        dist.barrier()

    end = time.perf_counter()
    elapsed_ms = (end - start) / num_timed_iterations * 1000

    if rank == 0:
        print(f"\n[Rank {rank}] Average latency per iteration: {elapsed_ms:.3f} ms")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
