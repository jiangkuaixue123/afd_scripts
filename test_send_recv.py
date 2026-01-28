# import os
# import torch
# import torch.distributed as dist
# from torch.profiler import profile, record_function, ProfilerActivity


# def test_func(x, rank):
#     if rank == 0:
#         x += 1
#         # Send the tensor to process 1
#         dist.send(tensor=x, dst=1)
#     else:
#         # Receive tensor from process 0
#         dist.recv(tensor=x, src=0)
#     return x + 2

# def run(rank):
#     torch.cuda.set_device(rank)
#     x = torch.ones([1024, 1024], device='cuda')
#     y = test_func(x, rank)
#     dist.barrier()
#     graph = torch.cuda.CUDAGraph()

#     with torch.cuda.graph(graph):
#         x = torch.ones([1024, 1024], device='cuda')
#         y = test_func(x, rank)

    
#     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#                 record_shapes=True,
#                 with_stack=True,
#                 on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./send_recv_graph_{rank}')) as prof:
#         for i in range(10):
#             x.copy_(torch.ones([1024, 1024], device='cuda'))
#             graph.replay()
#     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
#     print(f"Rank{rank} has data {y}")


# def main():
#     rank = int(os.environ['RANK'])
#     local_rank = int(os.environ['LOCAL_RANK'])
#     world_size = int(os.environ['WORLD_SIZE'])
#     dist.init_process_group('nccl', rank=rank, world_size=world_size)
#     run(local_rank)

# if __name__ == "__main__":
#     main()


import os
import torch
import torch.distributed as dist

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29500",
        rank=local_rank,
        world_size=4
    )

    # 定义角色：rank0为主节点，其余为从节点
    # 注意：dist.new_group 是一个集合操作，所有进程必须以相同的顺序调用它，
    # 即使该进程不在新建的组中。
    groups = {}
    worker_ranks = [1, 2, 3]
    
    print(f"Rank{local_rank} starting to create process groups...")
    for worker_rank in worker_ranks:
        # 所有进程都必须执行这行代码
        group = dist.new_group(ranks=[0, worker_rank])
        print(f"Rank{local_rank} is creating process group {group}\n")
        
        # 保存对自己有用的组
        if local_rank == 0:
            groups[worker_rank] = group
        elif local_rank == worker_rank:
            groups[0] = group 
            
    print(f"Rank{local_rank} finished creating process groups.")

    if local_rank == 0:
        # 主节点与所有从节点分别通信
        for worker_rank in worker_ranks:
            print(f"Rank{local_rank} is sending message to Rank{worker_rank}\n")
            sub_group = groups[worker_rank]
            
            # 主节点给从节点发送指令
            msg = torch.tensor([1], dtype=torch.int)
            dist.send(msg, dst=worker_rank, group=sub_group)
            print(f"主节点(Rank0) | 给Rank{worker_rank}发送指令")
    else:
        # 从节点接收主节点指令
        if 0 in groups: # 确保当前节点有对应的通信组
            print(f"Rank{local_rank} is receiving message from Rank0\n")
            sub_group = groups[0]
            
            msg = torch.tensor([0], dtype=torch.int)
            dist.recv(msg, src=0, group=sub_group)
            print(f"从节点(Rank{local_rank}) | 接收主节点指令: {msg.item()}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()