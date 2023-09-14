import numpy as np
import torch
import torch.nn as nn
import os
import wandb
from unet import UNet
from PIL import Image

from Ddataset import get_dataloader


def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    """build target for dice coefficient"""
    dice_target = target.clone()
    if ignore_index >= 0:
        ignore_mask = torch.eq(target, ignore_index)
        dice_target[ignore_mask] = 0
        # [N, H, W] -> [N, H, W, C]
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
        dice_target[ignore_mask] = ignore_index
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    return dice_target.permute(0, 3, 1, 2)


def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    # 计算一个batch中所有图片某个类别的dice_coefficient
    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1)
        if ignore_index >= 0:
            # 找出mask中不为ignore_index的区域
            roi_mask = torch.ne(t_i, ignore_index)
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        if sets_sum == 0:
            sets_sum = 2 * inter

        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    return d / batch_size


def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """Average of Dice coefficient for all classes"""
    dice = 0.
    for channel in range(x.shape[1]):
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)

    return dice / x.shape[1]


def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
    # Dice loss (objective to minimize) between 0 and 1
    x = nn.functional.softmax(x, dim=1)
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index)

def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    losses = {}
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
        if dice is True:
            dice_target = build_target(target, num_classes, ignore_index)
            loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
        losses[name] = loss

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']



if __name__ == '__main__':
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    lr = 0.001
    epoch = 150
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    num_classes = 2

    print(torch.device)
    model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    model.load_state_dict(torch.load('./best_model.pth')['model'])
    train_loader, test_loader = get_dataloader(batch_size=1)

    roi_img = Image.open('./DRIVE/training/mask/21_training_mask.gif').convert('L')
    roi_img = np.array(roi_img)
    print(roi_img.shape)

    for inputs, labels in train_loader:
        outputs = model(inputs)['out']
        print('outputs.shape:', outputs.shape)
        outputs = outputs.argmax(1).squeeze(0)
        print('outputs.shape:', outputs.shape)
        outputs = outputs.to("cpu").numpy().astype(np.uint8)
        outputs[outputs == 1] = 255
        # outputs[roi_img == 0] = 0
        mask = Image.fromarray(outputs)
        mask.show()

        print('labels.shape:', labels.shape)

