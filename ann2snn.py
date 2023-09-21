import numpy as np
import torch
import torch.nn as nn
from main import criterion
import os
from unet import UNet
from spiking_unet import S_UNet
from spikingjelly.activation_based import functional, ann2snn
from PIL import Image

from Ddataset import get_dataloader


if __name__ == '__main__':
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    lr = 0.0001
    epoch = 150
    batch_size = 2
    T = 2
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    num_classes = 2

    train_dataloader, test_dataloader = get_dataloader(batch_size=batch_size)

    model_converter = ann2snn.Converter(mode='max', dataloader=train_dataloader)
    model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    model.to(device)
    snn_model = model_converter(model)
    snn_model.to(device)
    print(snn_model)

    loss_weight = torch.as_tensor([1.0, 2.0], device=device)

    losses = []
    for x, y in train_dataloader:
        x = x.repeat(T, 1, 1, 1, 1)
        x = x.to(device)
        y = y.to(device)
        outputs = functional.multi_step_forward(x, snn_model)
        outputs = outputs.mean(0)
        loss = criterion(outputs, y, loss_weight, dice=True, num_classes=num_classes, ignore_index=255)
        losses.append(loss.item())
        functional.reset_net(snn_model)
    print(sum(losses) / len(losses))

