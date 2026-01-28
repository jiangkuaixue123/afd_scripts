import torch

torch._logging.set_logs(graph_code=True)

def bar(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b


opt_bar = torch.compile(bar, fullgraph=True)
inp1 = torch.ones(10)
inp2 = torch.ones(10)
print(opt_bar(inp1, inp2))
print(opt_bar(inp1, -inp2))