import torch
import numpy


a = torch.load('./checkpoint_latest.pth')
b = torch.load('./checkpoint_max.pth')
print(a['net'])