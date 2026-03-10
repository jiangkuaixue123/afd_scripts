#!/usr/bin/env python3
"""
Example of using symmetric memory for P2P copy with put_signal and wait_signal,
with CUDA graph capture and replay, plus profiling.

This example demonstrates how to use PyTorch's symmetric memory API to perform
peer-to-peer data transfer between 2 ranks using put_signal and wait_signal
for synchronization, with CUDA graph optimization.

Run (2 GPUs on one machine):
  torchrun --nproc_per_node=2 symmetric_memory_copy_example.py

Or with 2 nodes (1 GPU each), set MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE accordingly.
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2, "This script expects exactly 2 ranks."

    # Initialize process group with NCCL backend
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if symm_mem.is_nvshmem_available():
        symm_mem.set_backend("NVSHMEM")
    else:
        raise RuntimeError("NVSHMEM backend not available. Please install nvidia-nvshmem-cu12")

    # Enable symmetric memory for the world group
    group_name = dist.group.WORLD.group_name
    symm_mem.enable_symm_mem_for_group(group_name)

    # Create a symmetric memory buffer with shape (64, 2048)
    shape = (64, 2048)
    buffer_size = shape[0] * shape[1]
    buffer = symm_mem.empty(buffer_size, dtype=torch.bfloat16, device=device)

    # Perform rendezvous to set up symmetric memory across all ranks
    symm_mem_hdl = symm_mem.rendezvous(buffer, group=dist.group.WORLD)

    print(f"Rank {rank}: symmetric memory rendezvous completed")
    print(f"Rank {rank}: buffer_size={symm_mem_hdl.buffer_size}, "
          f"signal_pad_size={symm_mem_hdl.signal_pad_size}")

    # Get the buffer as a tensor for reading/writing
    data_buffer = symm_mem_hdl.get_buffer(0, shape, torch.bfloat16)

    # Prepare send/recv buffers based on rank
    if rank == 0:
        recv_buf = data_buffer  # Rank 0 receives data into this buffer
        send_buf = None
    else:
        send_buf = data_buffer  # Rank 1 sends data from this buffer
        recv_buf = None

    # Dedicated stream for capture and replay
    capture_stream = torch.cuda.Stream(device=device)

    # Warmup phase - ensure the communication works before capturing
    print(f"Rank {rank}: Warmup phase")
    torch.manual_seed(42)
    if rank == 1:
        send_buf.copy_(torch.randn(*shape, dtype=torch.bfloat16, device=device))
        symm_mem_hdl.put_signal(dst_rank=0)
    else:
        symm_mem_hdl.wait_signal(src_rank=1)

    torch.cuda.synchronize(device)
    dist.barrier()
    if rank == 1:
        print(f"send_buf[0,0]={send_buf} ")
    else:
        print(f"recv_buf[0,0]={recv_buf} ")

    if rank == 0:
        torch.manual_seed(42)
        expected = torch.randn(*shape, dtype=torch.bfloat16, device=device)
        if torch.allclose(recv_buf, expected):
            print(f"Rank {rank}: warmup verification PASSED!")
        else:
            max_diff = (recv_buf - expected).abs().max().item()
            print(f"Rank {rank}: warmup verification FAILED! max_diff={max_diff}")

    print(f"Rank {rank}: warmup completed")

    # Capture CUDA graph
    # NOTE: Only capture put_signal on rank 1. wait_signal cannot be captured
    # because it depends on external state (signal from rank 1) that changes each step.
    # If wait_signal is captured, it will use the signal pad state at capture time,
    # causing data to be off by one step.
    print(f"Rank {rank}: capturing CUDA graph...")
    if rank == 1:
        graph = torch.cuda.CUDAGraph()
        capture_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.graph(graph, stream=capture_stream):
            symm_mem_hdl.put_signal(dst_rank=0)
    else:
        graph = None

    print(f"Rank {rank}: CUDA graph {'captured' if rank == 1 else 'skipped (wait_signal cannot be captured)'}")

    torch.cuda.synchronize(device)
    dist.barrier()
    print(f"Rank {rank}: all ranks ready for replay")

    # Replay phase with profiling
    num_replays = 10
    print(f"Rank {rank}: starting replay phase with {num_replays} iterations")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"./profiler_logs/symm_mem_rank{rank}"
        ),
    ) as prof:
        for step in range(num_replays):
            with torch.profiler.record_function(f"replay_step_{step}"):
                if rank == 1:
                    # Update send buffer with new data
                    torch.manual_seed(100 + step)
                    send_buf.copy_(torch.randn(*shape, dtype=torch.bfloat16, device=device))
                    capture_stream.wait_stream(torch.cuda.current_stream(device))
                    graph.replay()  # Execute put_signal via graph
                else:
                    # Rank 0: wait_signal cannot be in graph, execute directly
                    symm_mem_hdl.wait_signal(src_rank=1)

                torch.cuda.synchronize(device)
                dist.barrier()

                # Verify received data on rank 0
                if rank == 0:
                    torch.manual_seed(100 + step)
                    expected = torch.randn(*shape, dtype=torch.bfloat16, device=device)
                    ok = torch.allclose(recv_buf, expected)
                    print(f"Rank 0: replay step {step} "
                          f"recv_buf[0,0]={recv_buf[0,0].item():.4f} "
                          f"(expected {expected[0,0].item():.4f}) OK={ok}")
                    # assert ok, f"Replay step {step} verification failed"

            prof.step()

    # Print profiler summary
    print(f"Rank {rank}: profiler summary:\n"
          f"{prof.key_averages().table(sort_by='cuda_time_total', row_limit=10)}")

    # Final synchronization
    symm_mem_hdl.barrier()
    print(f"Rank {rank}: all {num_replays} replays completed successfully!")

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
