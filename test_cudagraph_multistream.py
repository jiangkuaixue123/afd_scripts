import torch
import time
import os
from torch.profiler import profile, record_function, ProfilerActivity

def main():
    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU.")
        return

    device = torch.device("cuda:0")
    
    # 准备数据
    # 减小数据量，避免单个 Kernel 占满 GPU 导致物理串行
    # 使用 1024 x 1024，这在现代 GPU 上足够小，可以并行执行
    N = 1024 * 1024
    a = torch.randn(N, device=device)
    b = torch.randn(N, device=device)
    c = torch.zeros(N, device=device)
    d = torch.zeros(N, device=device)

    # 创建流和事件
    stream1 = torch.cuda.Stream(device=device)
    stream2 = torch.cuda.Stream(device=device)
    event1 = torch.cuda.Event()
    event2 = torch.cuda.Event()

    # 定义要在图中运行的函数
    def workload(a, b, c, d):
        # Stream 1: 密集的 Pointwise 计算链
        with torch.cuda.stream(stream1):
            for _ in range(50): # 增加 Kernel 数量
                torch.sin(a, out=c)
                torch.cos(c, out=c)
            event1.record(stream1)

        # Stream 2: 密集的 Pointwise 计算链
        with torch.cuda.stream(stream2):
            # 1. 【并行部分】
            # 这些小 Kernel 应该能和 Stream 1 的 Kernel 明显重叠
            for _ in range(50):
                torch.exp(b, out=d)
                # 避免链式调用产生中间 Tensor，改用显式的 out= 写法
                # d = |d|
                torch.abs(d, out=d)
                # d = d + 1
                torch.add(d, 1.0, out=d)
                # d = log(d)
                torch.log(d, out=d)
            
            # 2. 【同步点】
            stream2.wait_event(event1)
            
            # 3. 【串行部分】
            torch.add(d, 1.0, out=d)
            
            event2.record(stream2)
        
        # Join
        stream1.wait_event(event2)

    print("Warmup...")
    # Warmup (不捕获)
    # 注意：为了让 graph capture 成功，warmup 最好也在相同的流模式下运行，或者至少让内存分配稳定
    for _ in range(3):
        workload(a, b, c, d)
    torch.cuda.synchronize()

    print("Capturing CUDA Graph...")
    
    # 创建 Graph 对象
    g = torch.cuda.CUDAGraph()

    # 在捕获之前，确保所有流都空闲
    torch.cuda.synchronize()

    # 开始捕获
    # 注意：capture_stream 通常指定为主流，其他流的操作也会被捕获
    with torch.cuda.graph(g, stream=stream1):
        workload(a, b, c, d)

    print("Graph captured successfully.")

    # 验证输出
    # 重置输出
    c.fill_(0)
    d.fill_(0)
    
    print("Replaying Graph...")
    
    log_dir = "./log/multistream_trace"
    os.makedirs(log_dir, exist_ok=True)
    
    # Replay with Profiler
    print(f"Profiling graph replay (saving to {log_dir})...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir)
    ) as prof:
        for i in range(20): # Replay 20 times
            with record_function("graph_replay_step"):
                g.replay()
    
    torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # 性能测试对比
    print("\nBenchmarking...")
    iterations = 10
    
    # Eager Mode
    log_dir_eager = "./log/multistream_eager_trace"
    os.makedirs(log_dir_eager, exist_ok=True)
    print(f"Profiling eager mode (saving to {log_dir_eager})...")

    torch.cuda.synchronize()
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir_eager)
    ) as prof_eager:
        for i in range(20):
             with record_function("eager_step"):
                workload(a, b, c, d)
    
    torch.cuda.synchronize()
    print(prof_eager.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Benchmark Loop (Timing only)
    start = time.time()
    for _ in range(iterations):
        workload(a, b, c, d)

    # Graph Mode
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        g.replay()
    torch.cuda.synchronize()
    graph_time = (time.time() - start) * 1000 / iterations
    print(f"Graph mode time: {graph_time:.3f} ms")
    print(f"Speedup: {eager_time / graph_time:.2f}x")

if __name__ == "__main__":
    main()
