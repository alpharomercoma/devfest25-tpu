import torch
import torch_xla

dev = torch_xla.device()

t1 = torch.tensor(
    [[1,2],[3,4]],
    dtype=torch.bfloat16,
    device=dev)
t2 = torch.tensor([
    [5,6],[7,8]],
    dtype=torch.bfloat16,
    device=dev)

print(t1 + t2) # element-wise addition
print(t1 @ t2) # matrix multiplication
print(t1 * t2) # element-wise multiplication
