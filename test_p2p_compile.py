import os
import torch
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity

def setup():
    # 初始化分布式环境
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

# 定义 send/recv 逻辑
# 使用 allow_in_graph 告诉 Dynamo 不要尝试 trace 这些分布式操作的内部实现
# 而是将它们作为图中的一个节点（BlackBox）
@torch.compiler.allow_in_graph
def dist_send(tensor, dst):
    dist.send(tensor, dst=dst)

@torch.compiler.allow_in_graph
def dist_recv(tensor, src):
    dist.recv(tensor, src=src)

# 使用 torch.compile 进行全图编译
# 注意：当外部使用 CUDA Graph 手动捕获时，compile 的 mode 可能会有影响。
# 这里我们保持 fullgraph=True，让 Dynamo 尝试捕获整个图。
@torch.compile(fullgraph=False)
def p2p_step(tensor: torch.Tensor, rank: int, peer_rank: int):
    # 注意：在编译模式下，条件分支最好基于常量或专门化的参数
    # 这里 rank 和 peer_rank 是 int，Dynamo 会特化编译
    if rank == 0:
        dist_send(tensor, dst=peer_rank)
    elif rank == 1:
        dist_recv(tensor, src=peer_rank)
    return tensor

def main():
    rank = setup()
    world_size = dist.get_world_size()

    if world_size < 2:
        if rank == 0:
            print("Error: This script requires at least 2 ranks.")
        cleanup()
        return

    device = torch.device(f"cuda:{rank}")
    
    # 准备数据
    # 100MB float32: 25 * 1024 * 1024 elements * 4 bytes/element = 100MB
    data_size = 25 * 1024 * 1024 
    tensor = torch.arange(data_size, dtype=torch.float32, device=device)
    
    peer_rank = 1 if rank == 0 else 0
    
    print(f"Rank {rank}: Preparing to compile and run...")

    # Warmup & Compile
    # 前几次运行会触发编译
    # 为了配合 CUDA Graph，我们先在默认流上跑几次确保编译完成
    for i in range(5):
        p2p_step(tensor, rank, peer_rank)
    
    torch.cuda.synchronize()
    if rank == 0:
        print("Warmup & Compilation finished.")

    # --- CUDA Graph Capture ---
    # 使用私有 stream 进行 capture 预热和录制
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    
    if rank == 0:
        print("Starting CUDA Graph capture...")

    # 1. Capture Warmup (在目标 stream 上运行几次)
    with torch.cuda.stream(s):
        for _ in range(3):
            p2p_step(tensor, rank, peer_rank)
    
    torch.cuda.current_stream().wait_stream(s)
    
    # 2. Capture
    g = torch.cuda.CUDAGraph()
    
    # 确保之前的操作完成
    torch.cuda.synchronize()
    
    with torch.cuda.graph(g):
        p2p_step(tensor, rank, peer_rank)
        
    if rank == 0:
        print("CUDA Graph captured successfully.")

    # --- Replay with Profiler ---
    # 验证数据前先重置一下，确保 replay 真的能在 rank 1 写入数据
    if rank == 1:
        tensor.fill_(0.0)
    elif rank == 0:
        tensor.fill_(42.0) # 发送 42
    
    torch.cuda.synchronize() # 确保数据准备好

    log_dir = f"./log/p2p_trace_rank{rank}"
    os.makedirs(log_dir, exist_ok=True)

    if rank == 0:
        print("Starting Replay with Profiler...")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir)
    ) as prof:
        for i in range(10):
            with record_function("graph_replay_step"):
                g.replay()
                # 注意：CUDA Graph replay 是异步的，profile 记录的是 CPU 端的 replay 发射时间
                # 以及 GPU 端的实际执行时间。
    
    torch.cuda.synchronize()
    
    if rank == 0:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print(f"Profiling finished. Logs saved to {log_dir}")

    # 验证数据
    if rank == 1:
        # Rank 1 应该收到 42.0
        if torch.all(tensor == 42.0):
            print(f"Rank {rank}: Success! Received correct data via CUDA Graph replay.")
        else:
            print(f"Rank {rank}: Failed! Data mismatch.")
            print(f"First few elements: {tensor[:10]}")

    cleanup()

if __name__ == "__main__":
    main()
