import torch.nn as nn
import torch
import numpy as np


def torch_bn1d_demo():
    N, D = 3, 5
    input = torch.arange(N * D).reshape(N, D).float()
    print(input)

    # bn_torch = nn.BatchNorm1d(D, affine=True, momentum=.5)
    bn1d = nn.BatchNorm1d(D, affine=True)  # enable weight and bias
    print(bn1d)
    print(bn1d.state_dict())
    # print(bn_torch.weight)
    # print(bn_torch.bias)

    output = bn1d(input)
    print(output)

    print(bn1d.state_dict())



def torch_bn2d_demo():
    N, C, H, W = 3, 2, 4, 5

    input = torch.arange(N * C * H * W).reshape(N, C, H, W).float()

    print(input)

    # bn_torch = nn.BatchNorm1d(D, affine=True, momentum=.5)
    bn2d = nn.BatchNorm2d(C, affine=True)  # enable weight and bias

    # print(bn2d.weight)
    # print(bn2d.bias)

    output = bn2d(input)
    print(output)

torch_bn1d_demo()