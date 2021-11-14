
import torch
import torch.nn as nn

input = torch.arange(2*3*4*5).reshape(2, 3, 4, 5).float()


batch_norm = nn.BatchNorm2d(3,affine=True)

output = batch_norm(input)

print(batch_norm.weight)
print(batch_norm.bias)