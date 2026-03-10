#!/usr/bin/env python3
"""
Example of using NVSHMEM put/get in Triton kernel for cross-node P2P communication.

This example demonstrates how to use NVSHMEM's device-side put/get APIs in Triton
kernels to perform cross-node peer-to-peer data transfer.

Key differences from the CUDA backend approach:
- NVSHMEM uses put/get APIs instead of direct pointer access for cross-node
- nvshmem_ptr() returns nullptr for remote peers, so get_buffer() won't work
- Communication happens inside Triton kernels using nvshmem.put() / nvshmem.get()

Requirements:
  - H100+ GPU (NVSHMEM requires H100)
  - NVSHMEM library installed (pip install nvidia-nvshmem-cu12)
  - RDMA network (InfiniBand or RoCE) for cross-node communication

Run (2 GPUs on one machine for testing):
  torchrun --nproc_per_node=2 symmetric_memory_nvshmem_cross_node.py

Run across 2 nodes (1 GPU each):
  # Node 0:
  MASTER_ADDR=<node0_ip> MASTER_PORT=29500 RANK=0 WORLD_SIZE=2 \
    python symmetric_memory_nvshmem_cross_node.py

  # Node 1:
  MASTER_ADDR=<node0_ip> MASTER_PORT=29500 RANK=1 WORLD_SIZE=2 \
    python symmetric_memory_nvshmem_cross_node.py
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

# Check if NVSHMEM is available before importing NVSHMEM-specific modules
if not symm_mem.is_nvshmem_available():
    print("ERROR: NVSHMEM not available. Please install nvidia-nvshmem-cu12")
    print("       pip install nvidia-nvshmem-cu12")
    sys.exit(1)

import triton
import triton.language as tl

import torch.distributed._symmetric_memory._nvshmem_triton as nvshmem
from torch.distributed._symmetric_memory._nvshmem_triton import requires_nvshmem


# =============================================================================
# Triton Kernels using NVSHMEM put/get for cross-node communication
# =============================================================================


@requires_nvshmem
@triton.jit
def nvshmem_put_kernel(
    dest_ptr,  # Destination buffer pointer (on remote peer)
    src_ptr,  # Source buffer pointer (local)
    nelems,  # Number of elements to transfer
    peer,  # Target peer rank
):
    """
    NVSHMEM put kernel: transfers data from local buffer to remote peer's buffer.
    This is a one-sided operation - the remote peer does not need to participate.
    """
    nvshmem.put(dest_ptr, src_ptr, nelems, peer)


@requires_nvshmem
@triton.jit
def nvshmem_get_kernel(
    dest_ptr,  # Destination buffer pointer (local)
    src_ptr,  # Source buffer pointer (on remote peer)
    nelems,  # Number of elements to transfer
    peer,  # Source peer rank
):
    """
    NVSHMEM get kernel: fetches data from remote peer's buffer to local buffer.
    This is a one-sided operation - the remote peer does not need to participate.
    """
    nvshmem.get(dest_ptr, src_ptr, nelems, peer)


@requires_nvshmem
@triton.jit
def nvshmem_put_signal_kernel(
    dest_ptr,  # Destination buffer pointer (on remote peer)
    src_ptr,  # Source buffer pointer (local)
    nelems,  # Number of elements to transfer
    signal_ptr,  # Signal pointer (on remote peer)
    signal_val,  # Signal value to write after put completes
    peer,  # Target peer rank
):
    """
    NVSHMEM put with signaling: transfers data and then writes a signal value.
    The remote peer can wait on this signal to know the transfer is complete.
    """
    # Perform the put operation
    nvshmem.put(dest_ptr, src_ptr, nelems, peer)
    # Ensure the put is complete before signaling
    nvshmem.quiet()
    # Use signal_op to set the signal value on remote peer
    # NVSHMEM_SIGNAL_SET = 0 (set signal to the given value)
    nvshmem.signal_op(signal_ptr, signal_val, 0, peer)


@requires_nvshmem
@triton.jit
def nvshmem_wait_signal_kernel(
    signal_ptr,  # Signal pointer to wait on
    expected_val,  # Expected signal value
):
    """
    Wait for a signal to reach the expected value.
    This blocks the kernel until the signal is received.
    """
    # NVSHMEM_SIGNAL_EQ = 0 (compare if signal == expected_val)
    nvshmem.signal_wait_until(signal_ptr, 0, expected_val)


@requires_nvshmem
@triton.jit
def nvshmem_putmem_signal_block_kernel(
    dst,  # Destination buffer (on remote peer)
    src,  # Source buffer (local)
    size_bytes,  # Size in bytes to transfer
    signal,  # Signal address (on remote peer)
    sig_val,  # Signal value to write
    sig_op,  # Signal operation (0=SET, 1=ADD, 2=AND, 3=OR, 4=XOR)
    peer,  # Target peer rank
):
    """
    Combined put and signal operation - more efficient than separate put + signal.
    The signal is written atomically after the data transfer.
    """
    nvshmem.putmem_signal_block(dst, src, size_bytes, signal, sig_val, sig_op, peer)


# =============================================================================
# Main Example
# =============================================================================


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2, "This example expects exactly 2 ranks"

    # Initialize process group with device_id to avoid warnings
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        device_id=device,
    )

    print(f"[Rank {rank}] Initializing NVSHMEM backend...")

    if symm_mem.is_nvshmem_available():
        symm_mem.set_backend("NVSHMEM")
    else:
        raise RuntimeError("NVSHMEM backend not available. Please install nvidia-nvshmem-cu12")

    # Enable symmetric memory for the world group
    group_name = dist.group.WORLD.group_name
    symm_mem.enable_symm_mem_for_group(group_name)

    # Configuration
    shape = (64, 2048)  # Data shape
    nelems = shape[0] * shape[1]  # Total number of elements
    dtype = torch.bfloat16

    print(f"[Rank {rank}] Creating symmetric memory buffers...")

    # Create symmetric memory buffers
    # Each rank has its own local buffer that can be accessed by remote peers
    data_buffer = symm_mem.empty(nelems, dtype=dtype, device=device)
    signal_buffer = symm_mem.empty(1, dtype=torch.int64, device=device)

    # Rendezvous to establish symmetric memory mappings
    symm_mem.rendezvous(data_buffer, group=dist.group.WORLD)
    symm_mem.rendezvous(signal_buffer, group=dist.group.WORLD)

    print(f"[Rank {rank}] Symmetric memory rendezvous completed")

    # Initialize data
    torch.manual_seed(42 + rank)
    if rank == 1:
        # Sender: fill buffer with data
        data_buffer.copy_(torch.randn(*shape, dtype=dtype, device=device).flatten())
        print(f"[Rank {rank}] Initialized send buffer with random data")

    # Clear signal
    signal_buffer.zero_()

    # =========================================================================
    # Example 1: One-sided PUT (Rank 1 pushes data to Rank 0)
    # =========================================================================
    print(f"\n[Rank {rank}] === Example 1: One-sided PUT ===")

    dist.barrier()

    if rank == 1:
        # Sender: push data to rank 0 using NVSHMEM put
        print(f"[Rank {rank}] Sending data to rank 0 using NVSHMEM put...")
        nvshmem_put_kernel[(1,)](
            data_buffer,  # Destination (on rank 0)
            data_buffer,  # Source (local on rank 1)
            nelems,
            0,  # Target peer: rank 0
        )
        print(f"[Rank {rank}] Sending data to rank 0 using NVSHMEM put finish")

    # Wait for transfer to complete
    dist.barrier()

    if rank == 0:
        # Receiver: verify data received
        torch.manual_seed(43)  # seed = 42 + 1 (rank 1's seed)
        expected = torch.randn(*shape, dtype=dtype, device=device).flatten()
        if torch.allclose(data_buffer, expected, rtol=1e-2, atol=1e-2):
            print(f"[Rank {rank}] Example 1 PASSED: Received data correctly!")
        else:
            max_diff = (data_buffer - expected).abs().max().item()
            print(f"[Rank {rank}] Example 1 FAILED: max_diff={max_diff}")

    # =========================================================================
    # Example 2: One-sided GET (Rank 0 pulls data from Rank 1)
    # =========================================================================
    print(f"\n[Rank {rank}] === Example 2: One-sided GET ===")

    # Reset buffers
    if rank == 0:
        data_buffer.zero_()
    if rank == 1:
        torch.manual_seed(100 + rank)
        data_buffer.copy_(torch.randn(*shape, dtype=dtype, device=device).flatten())

    dist.barrier()

    if rank == 0:
        # Receiver: pull data from rank 1 using NVSHMEM get
        print(f"[Rank {rank}] Getting data from rank 1 using NVSHMEM get...")
        nvshmem_get_kernel[(1,)](
            data_buffer,  # Destination (local on rank 0)
            data_buffer,  # Source (on rank 1)
            nelems,
            1,  # Source peer: rank 1
        )

    dist.barrier()

    if rank == 0:
        torch.manual_seed(101)  # seed = 100 + 1 (rank 1's seed)
        expected = torch.randn(*shape, dtype=dtype, device=device).flatten()
        if torch.allclose(data_buffer, expected, rtol=1e-2, atol=1e-2):
            print(f"[Rank {rank}] Example 2 PASSED: Received data correctly!")
        else:
            max_diff = (data_buffer - expected).abs().max().item()
            print(f"[Rank {rank}] Example 2 FAILED: max_diff={max_diff}")

    # =========================================================================
    # Example 3: PUT with signal synchronization
    # =========================================================================
    print(f"\n[Rank {rank}] === Example 3: PUT with Signal ===")

    # Reset
    if rank == 1:
        torch.manual_seed(200 + rank)
        data_buffer.copy_(torch.randn(*shape, dtype=dtype, device=device).flatten())
    if rank == 0:
        data_buffer.zero_()
    signal_buffer.zero_()

    dist.barrier()

    if rank == 1:
        # Sender: put data and signal
        print(f"[Rank {rank}] Sending data with signal to rank 0...")
        nvshmem_put_signal_kernel[(1,)](
            data_buffer,
            data_buffer,
            nelems,
            signal_buffer,
            12345,  # Signal value
            0,  # Target peer: rank 0
        )

    if rank == 0:
        # Receiver: wait for signal then read data
        print(f"[Rank {rank}] Waiting for signal...")
        nvshmem_wait_signal_kernel[(1,)](
            signal_buffer,
            12345,  # Expected signal value
        )
        print(f"[Rank {rank}] Signal received, verifying data...")

        torch.manual_seed(201)
        expected = torch.randn(*shape, dtype=dtype, device=device).flatten()
        if torch.allclose(data_buffer, expected, rtol=1e-2, atol=1e-2):
            print(f"[Rank {rank}] Example 3 PASSED!")
        else:
            max_diff = (data_buffer - expected).abs().max().item()
            print(f"[Rank {rank}] Example 3 FAILED: max_diff={max_diff}")

    dist.barrier()

    # =========================================================================
    # Example 4: putmem_signal_block (atomic put + signal)
    # =========================================================================
    print(f"\n[Rank {rank}] === Example 4: Atomic putmem_signal_block ===")

    # Reset
    if rank == 1:
        torch.manual_seed(300 + rank)
        data_buffer.copy_(torch.randn(*shape, dtype=dtype, device=device).flatten())
    if rank == 0:
        data_buffer.zero_()
    signal_buffer.zero_()

    dist.barrier()

    if rank == 1:
        # Use putmem_signal_block for atomic put + signal
        print(f"[Rank {rank}] Using putmem_signal_block...")
        nbytes = nelems * data_buffer.element_size()
        nvshmem_putmem_signal_block_kernel[(1,)](
            data_buffer,  # Destination (on rank 0)
            data_buffer,  # Source (local)
            nbytes,  # Size in bytes
            signal_buffer,  # Signal address (on rank 0)
            99999,  # Signal value
            0,  # Signal operation: SET
            0,  # Target peer: rank 0
        )

    if rank == 0:
        # Wait for signal
        print(f"[Rank {rank}] Waiting for atomic signal...")
        nvshmem_wait_signal_kernel[(1,)](
            signal_buffer,
            99999,
        )
        print(f"[Rank {rank}] Atomic signal received!")

        torch.manual_seed(301)
        expected = torch.randn(*shape, dtype=dtype, device=device).flatten()
        if torch.allclose(data_buffer, expected, rtol=1e-2, atol=1e-2):
            print(f"[Rank {rank}] Example 4 PASSED!")
        else:
            max_diff = (data_buffer - expected).abs().max().item()
            print(f"[Rank {rank}] Example 4 FAILED: max_diff={max_diff}")

    dist.barrier()
    print(f"\n[Rank {rank}] All examples completed!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
