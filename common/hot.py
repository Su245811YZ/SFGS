import torch
import torchvision.transforms.functional as TF

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def load_image_as_tensor(path: str) -> torch.Tensor:
    image = Image.open(path).convert('RGB')
    image = TF.to_tensor(image)
    return image


def compute_psnr_map(img1: torch.Tensor, img2: torch.Tensor, eps=1e-8) -> torch.Tensor:
    mse_map = (img1 - img2).pow(2).mean(dim=0)
    psnr_map = 10 * torch.log10(1.0 / (mse_map + eps))
    return psnr_map


def visualize_psnr_heatmap(psnr_map: torch.Tensor, save_path: str, title='PSNR Heatmap', vmin=20, vmax=40):
    heatmap = psnr_map.detach().cpu().numpy()

    plt.figure(figsize=(6, 6))
    im = plt.imshow(heatmap, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='PSNR (dB)')
    plt.title(title)
    plt.axis('off')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


if __name__ == "__main__":
    gt_path = '/home/suyuze/workspace/ExAvatar_mine/0723bestall/result/00028/test/Take6/130_gt.png'
    rendered_path = '/home/suyuze/workspace/ExAvatar_mine/0723bestall/result/00028/test/Take6/130_human_refined.png'
    save_path ='/home/suyuze/workspace/ExAvatar_mine/130.png'

    gt_img = load_image_as_tensor(gt_path)
    rendered_img = load_image_as_tensor(rendered_path)

    assert gt_img.shape == rendered_img.shape, "图像尺寸不一致，请检查"

    psnr_map = compute_psnr_map(gt_img, rendered_img)
    visualize_psnr_heatmap(psnr_map, save_path=save_path, vmin=25, vmax=40)

    print(f"PSNR 热图已保存到: {save_path}")