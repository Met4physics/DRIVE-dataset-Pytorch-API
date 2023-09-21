import numpy as np
import torch
import torch.nn as nn
import os
from unet import UNet
from spiking_unet import S_UNet
import transform
from spikingjelly.activation_based import functional
from PIL import Image

from Ddataset import get_dataloader


if __name__ == '__main__':
    weights_path = "./checkpoint_max.pth"
    img_path = "./DRIVE/test/images/01_test.tif"
    roi_mask_path = "./DRIVE/test/mask/01_test_mask.gif"

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    num_classes = 2
    T = 6
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    s_model = S_UNet(in_channels=3, num_classes=num_classes, base_c=32, T=T)
    s_model.load_state_dict(torch.load(weights_path)['net'])
    best_epoch = torch.load(weights_path)['epoch']
    print(f'best epoch: {best_epoch}')
    s_model.to(device)
    roi_img = Image.open(roi_mask_path).convert('L')
    roi_img = np.array(roi_img)
    original_img = Image.open(img_path).convert('RGB')
    data_transform = transform.Compose([transform.ToTensor(), transform.Normalize(mean=mean, std=std)])
    img, img2 = data_transform(original_img, original_img)
    img = torch.unsqueeze(img, dim=0)

    s_model.eval()
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        s_model(init_img)

        output = s_model(img.to(device))

        prediction = output.argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # 将前景对应的像素值改成255(白色)
        prediction[prediction == 1] = 255
        # 将不敢兴趣的区域像素设置成0(黑色)
        prediction[roi_img == 0] = 0
        mask = Image.fromarray(prediction)
        mask.save("test_result.png")
        print('finish')