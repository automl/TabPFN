import torch
from torch import nn

def get_NormalInitializer(std):
    def initializer(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, std)
            nn.init.normal_(m.bias, 0, std)
    return initializer