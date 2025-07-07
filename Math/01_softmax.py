import torch
import torch.nn.functional as F

vector = torch.randn(10, dtype = torch.float32)
print("Input vector: ", vector)
output = F.softmax(vector, dim = -1)

print("Output vector: ", output)
