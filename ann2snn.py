import numpy as np
import torch
import torch.nn as nn
from main import criterion, ConfusionMatrix
import os
import transform
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
    T = 250
    resume = False
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    num_classes = 2

    train_dataloader, test_dataloader = get_dataloader(batch_size=batch_size)

    model_path = './checkpoint_max.pth'
    model_converter = ann2snn.Converter(mode='max', dataloader=train_dataloader)
    model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    model.load_state_dict(torch.load(model_path)['net'])
    model.to(device)
    snn_model = model_converter(model)
    snn_model.to(device)
    print(snn_model)

    loss_weight = torch.as_tensor([1.0, 2.0], device=device)

    losses = []

    conf_mat = ConfusionMatrix(num_classes)
    with torch.no_grad():
        for x, y in train_dataloader:
            x = x.repeat(T, 1, 1, 1, 1)
            x = x.to(device)
            y = y.to(device)
            outputs = functional.multi_step_forward(x, snn_model)
            outputs = outputs.mean(0)
            loss = criterion(outputs, y, loss_weight, dice=True, num_classes=num_classes, ignore_index=255)
            losses.append(loss.item())
            conf_mat.update(y.flatten(), outputs.argmax(1).flatten())
            functional.reset_net(snn_model)
    print('train:')
    print('loss:', round(sum(losses) / len(losses), 3))
    acc_global, acc, iou, f1 = conf_mat.compute()
    acc_global, acc, iou, f1 = round(acc_global.item(), 3), list(np.around(acc.cpu().numpy(), 3)), list(np.around(iou.cpu().numpy(), 3)), list(np.around(f1.cpu().numpy(), 3))
    print(f'acc_global: {acc_global}, acc: {acc}, iou: {iou}, f1: {f1}')

    losses.clear()
    conf_mat.reset()
    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.repeat(T, 1, 1, 1, 1)
            x = x.to(device)
            y = y.to(device)
            outputs = functional.multi_step_forward(x, snn_model)
            outputs = outputs.mean(0)
            loss = criterion(outputs, y, loss_weight, dice=True, num_classes=num_classes, ignore_index=255)
            losses.append(loss.item())
            conf_mat.update(y.flatten(), outputs.argmax(1).flatten())
            functional.reset_net(snn_model)
    print('test:')
    print('loss:', round(sum(losses) / len(losses), 3))
    acc_global, acc, iou, f1 = conf_mat.compute()
    acc_global, acc, iou, f1 = round(acc_global.item(), 3), list(np.around(acc.cpu().numpy(), 3)), list(np.around(iou.cpu().numpy(), 3)), list(np.around(f1.cpu().numpy(), 3))
    print(f'acc_global: {acc_global}, acc: {acc}, iou: {iou}, f1: {f1}')


    if resume:
        img_path = "./DRIVE/test/images/01_test.tif"
        roi_mask_path = "./DRIVE/test/mask/01_test_mask.gif"

        roi_img = Image.open(roi_mask_path).convert('L')
        roi_img = np.array(roi_img)
        original_img = Image.open(img_path).convert('RGB')
        data_transform = transform.Compose([transform.ToTensor(), transform.Normalize(mean=mean, std=std)])
        img, img2 = data_transform(original_img, original_img)
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            snn_model(init_img)

            img = img.to(device)
            img = img.repeat(T, 1, 1, 1, 1)
            output = functional.multi_step_forward(img, snn_model).mean(0)

            prediction = output.argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            # 将前景对应的像素值改成255(白色)
            prediction[prediction == 1] = 255
            # 将不敢兴趣的区域像素设置成0(黑色)
            prediction[roi_img == 0] = 0
            mask = Image.fromarray(prediction)
            mask.save("test_result.png")
            print('finish')
