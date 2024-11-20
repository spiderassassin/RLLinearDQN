import torch
import time
import torch

device = "mps"

torch.manual_seed(1234)
TENSOR_A_CPU = torch.rand(50, 50)
TENSOR_B_CPU = torch.rand(50, 50)

torch.manual_seed(1234)
TENSOR_A_MPS = torch.rand(50, 50).to(device)
TENSOR_B_MPS = torch.rand(50, 50).to(device)

start_time = time.time()
torch.matmul(TENSOR_A_CPU, TENSOR_B_CPU)
print("CPU : --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
torch.matmul(TENSOR_A_MPS, TENSOR_B_MPS)
print("MPS : --- %s seconds ---" % (time.time() - start_time))