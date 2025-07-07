import torch
import torch.nn.functional as F

vector = torch.randn(10, dtype = torch.float32)
print("Input vector: ", vector)
output = F.softmax(vector, dim = -1) # dim = -1 basically means that we are applying the operation among the last dimension

print("Output vector: ", output)
