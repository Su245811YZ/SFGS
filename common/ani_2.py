import argparse
import os
import os.path as osp
import torch
import numpy as np
import json
import cv2
from glob import glob
from tqdm import tqdm
from config import cfg
from base import Tester
from utils.smpl_x import smpl_x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--motion_path', type=str, dest='motion_path')
    parser.add_argument('--out_dir', type=str, default='render_out', help='Output directory')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg.set_args(args.subject_id)
    tester = Tester(args.test_epoch)

    # 1. 加载 ID 信息 (保持原框架)
    root_path = osp.join('..', 'data', cfg.dataset, 'data', cfg.subject_id)
    def load_json_tensor(name):
        with open(osp.join(root_path, 'smplx_optimized', name)) as f:
            return torch.FloatTensor(json.load(f))

    smpl_x.set_id_info(
        load_json_tensor('shape_param.json'),
        load_json_tensor('face_offset.json'),
        load_json_tensor('joint_offset.json'),
        load_json_tensor('locator_offset.json')
    )

    tester.smplx_params = None
    tester._make_model()

    # 2. 准备输出目录
    # 自动在 out_dir 下创建以 subject_id 命名的文件夹
    save_path = osp.join(args.out_dir, args.subject_id, 'raw_renders')
    os.makedirs(save_path, exist_ok=True)
    print(f"==> Frames will be saved to: {save_path}")

    # 3. 获取待处理序列
    frame_idx_list = sorted([int(x.split('/')[-1][:-5]) for x in glob(osp.join(args.motion_path, 'smplx_optimized', 'smplx_params_smoothed', '*.json'))])
    
    # 尝试获取渲染分辨率
    render_shape = (1024, 1024) 

    for frame_idx in tqdm(frame_idx_list):
        # 加载相机和模型参数
        with open(osp.join(args.motion_path, 'cam_params', f"{frame_idx}.json")) as f:
            cam_param = {k: torch.FloatTensor(v).cuda() for k,v in json.load(f).items()}
        with open(osp.join(args.motion_path, 'smplx_optimized', 'smplx_params_smoothed', f"{frame_idx}.json")) as f:
            smplx_param = {k: torch.FloatTensor(v).cuda().view(-1) for k,v in json.load(f).items()}
        
        t = torch.tensor(frame_idx).cuda() / 30 
        
        with torch.no_grad():
            # 获取 Gaussian 资产并渲染
            # human_asset 包含了模型在当前动作下的状态
            human_asset, _, _, _ = tester.model.module.human_gaussian(smplx_param, cam_param, True, t)
            
            # 执行渲染
            render_out = tester.model.module.gaussian_renderer(human_asset, render_shape, cam_param)
            
            # --- 直接提取图像内容 ---
            # render_out['img'] 形状通常为 (3, H, W)，值域 0-1
            img = render_out['img'].cpu().numpy().transpose(1, 2, 0) # 转换为 (H, W, 3)
            
            # 这里的 img 已经是渲染器生成的最终图（包含背景，如果有的话）
            # 我们只需要转为 BGR 格式并放大到 0-255
            final_frame = (img[:, :, ::-1] * 255).clip(0, 255).astype(np.uint8)
            
            # 保存
            save_name = f"{frame_idx:06d}_raw.png"
            cv2.imwrite(osp.join(save_path, save_name), final_frame)

if __name__ == "__main__":
    main()