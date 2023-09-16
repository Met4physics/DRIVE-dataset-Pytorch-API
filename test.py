import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, step_mode='m'):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.c = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        t = self.c(x)
        return t



class Down(nn.Module):
    def __init__(self, in_channels, out_channels, step_mode='m'):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, inputs):
        t = self.pool(inputs)
        tt = self.conv(t)
        return tt


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False, step_mode='m'):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [T, N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, num_classes, step_mode='m'):
        super(OutConv, self).__init__()
        self.c = nn.Conv2d(in_channels, num_classes, kernel_size=1)


    def forward(self, x):
        temp = self.c(x)
        return temp


class ts_UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = False,
                 base_c: int = 64,
                 step_mode: str = 'm',
                 T: int = 4):
        super(ts_UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.T = T

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2, step_mode)
        self.down2 = Down(base_c * 2, base_c * 4, step_mode)
        self.down3 = Down(base_c * 4, base_c * 8, step_mode)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor, step_mode)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear, step_mode)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear, step_mode)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear, step_mode)
        self.up4 = Up(base_c * 2, base_c, bilinear, step_mode)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return logits
