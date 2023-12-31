import numpy as np
import torch
import torch.nn as nn
import os
from spiking_unet import S_UNet
from spikingjelly.activation_based import functional
from PIL import Image
import wandb

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
    loss = nn.functional.cross_entropy(inputs, target, ignore_index=ignore_index, weight=loss_weight)
    if dice is True:
        dice_target = build_target(target, num_classes, ignore_index)
        loss += dice_loss(inputs, dice_target, multiclass=True, ignore_index=ignore_index)

    return loss


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            # 创建混淆矩阵
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            # 寻找GT中为目标的像素索引
            k = (a >= 0) & (a < n)
            # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
        acc_global = torch.diag(h).sum() / h.sum()
        # 计算每个类别的准确率
        acc = torch.diag(h) / h.sum(1)
        # 计算每个类别预测与真实目标的iou
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        recall = torch.diag(h) / h.sum(0)
        f1 = 2 * acc * recall / (acc + recall)
        return acc_global, acc, iu, f1


def set_seed(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    set_seed(3407)
    lr = 0.0001
    epoch = 150
    batch_size = 2
    T = 6
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    num_classes = 2

    wandb.init(
        project='spiking-unet',

        config={
            "learning_rate": lr,
            "epochs": epoch,
            "optimizer": "Adam",
            "T": T,
        }
    )

    print(torch.cuda.current_device())
    # model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    s_model = S_UNet(in_channels=3, num_classes=num_classes, base_c=32, T=T)
    # s_model = ts_UNet(in_channels=3, num_classes=num_classes, base_c=32)
    s_model = s_model.to(device)
    # model.load_state_dict(torch.load('./best_model.pth')['model'])
    train_loader, test_loader = get_dataloader(batch_size=batch_size)
    optimizer = torch.optim.Adam(s_model.parameters(), lr=lr)

    confmat = ConfusionMatrix(num_classes)
    step_confmat = ConfusionMatrix(num_classes)
    s_model.train()
    loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    min_loss = 100

    for i in range(epoch):
        l = []
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = s_model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, labels, loss_weight, dice=True, num_classes=num_classes, ignore_index=255)
            step_confmat.update(labels.flatten(), outputs.argmax(1).flatten())
            # loss = nn.functional.cross_entropy(outputs, labels, ignore_index=255, weight=loss_weight)
            wandb.log({'loss': loss})
            l.append(loss.item())
            loss.backward()
            optimizer.step()
            functional.reset_net(s_model)
        s_acc_global, s_acc, s_iou, s_f1 = step_confmat.compute()
        s_acc, s_iou, s_f1 = s_acc.mean().item(), s_iou.mean().item(), s_f1.mean().item()
        step_confmat.reset()
        l_mean = round(sum(l) / len(l), 3)
        wandb.log({'s_acc_global': s_acc_global, 's_acc': s_acc, 's_iou': s_iou, 's_f1': s_f1})
        print(f'epoch {i}: loss = {l_mean}')

        save_max = False
        if l_mean < min_loss:
            min_loss = l_mean
            save_max = True

        checkpoint = {
            'net': s_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'min_loss': min_loss
        }

        if save_max:
            torch.save(checkpoint, './checkpoint_max.pth')

        torch.save(checkpoint, './checkpoint_latest.pth')

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = s_model(inputs)
            confmat.update(labels.flatten(), outputs.argmax(1).flatten())
            functional.reset_net(s_model)

    acc_global, acc, iou, f1 = confmat.compute()
    acc_global, acc, iou, f1 = acc_global.item(), acc.tolist(), iou.tolist(), f1.tolist()
    print(f'acc_global: {acc_global}, acc: {acc}, iou: {iou}, f1: {f1}')
    acc_global, acc_mean, iou_mean, f1_mean = round(acc_global, 3), round(sum(acc) / len(acc), 3), round(sum(iou) / len(
        iou), 3), round(sum(f1) / len(f1), 3)
    print(f'mean acc_global: {acc_global}, mean acc: {acc_mean}, mean iou: {iou_mean}, mean f1: {f1_mean}')
