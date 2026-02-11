"""
CUDA Graph capture & replay 示例，并在 capture 后将图 debug_dump 到文件。
使用方式: python cudagraph_capture_replay.py
"""
import os
import torch


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU.")
        return

    device = torch.device("cuda:0")

    # 示例 workload：简单的前向计算
    N = 1024 * 1024
    a = torch.randn(N, device=device)
    b = torch.randn(N, device=device)
    c = torch.zeros(N, device=device)

    def workload(a, b, c):
        torch.add(a, b, out=c)
        torch.relu_(c)  # in-place, relu() 不支持 out=
        torch.mul(c, 2.0, out=c)

    # Warmup
    print("Warmup...")
    for _ in range(3):
        workload(a, b, c)
    torch.cuda.synchronize()

    # 创建 CUDA Graph。debug_dump 需要 keep_graph=True，否则 capture_end 后底层
    # cudaGraph_t 会被丢弃，debug_dump() 无法写入文件。
    g = torch.cuda.CUDAGraph(keep_graph=True)
    g.enable_debug_mode()

    torch.cuda.synchronize()
    print("Capturing CUDA Graph...")

    with torch.cuda.graph(g):
        workload(a, b, c)

    print("Graph captured successfully.")

    # 将图 dump 到文件（需在 enable_debug_mode() 且 capture 完成后调用）
    dump_dir = "./log/cudagraph_debug"
    os.makedirs(dump_dir, exist_ok=True)
    debug_path = os.path.join(dump_dir, "cudagraph.dot")
    g.debug_dump(debug_path)
    print(f"Graph debug_dump saved to: {debug_path}")

    # keep_graph=True 时需手动 instantiate 后再 replay（否则首次 replay 会隐式 instantiate，延迟较大）
    g.instantiate()

    # Replay
    print("Replaying graph 10 times...")
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    print("Replay done. Check output tensor (first 5):", c[:5].tolist())


if __name__ == "__main__":
    main()
