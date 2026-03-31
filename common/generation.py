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
from utils.vis import render_mesh

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--motion_path', type=str, dest='motion_path')
    parser.add_argument('--output_dir', type=str, default='render_results', help='output image folder')
    args = parser.parse_args()

    assert args.subject_id, "Please set subject ID"
    assert args.test_epoch, 'Test epoch is required.'
    assert args.motion_path, 'Motion path for the animation is required.'
    return args

def main():
    args = parse_args()
    cfg.set_args(args.subject_id)
    os.makedirs(args.output_dir, exist_ok=True)

    tester = Tester(args.test_epoch)

    # load ID information
    root_path = osp.join('..', 'data', cfg.dataset, 'data', cfg.subject_id)
    with open(osp.join(root_path, 'smplx_optimized', 'shape_param.json')) as f:
        shape_param = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, 'smplx_optimized', 'face_offset.json')) as f:
        face_offset = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, 'smplx_optimized', 'joint_offset.json')) as f:
        joint_offset = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, 'smplx_optimized', 'locator_offset.json')) as f:
        locator_offset = torch.FloatTensor(json.load(f))
    smpl_x.set_id_info(shape_param, face_offset, joint_offset, locator_offset)

    tester.smplx_params = None
    tester._make_model()

    motion_path = args.motion_path
    frame_idx_list = sorted([
        int(x.split('/')[-1][:-5]) 
        for x in glob(osp.join(args.motion_path, 'smplx_optimized', 'smplx_params_smoothed', '*.json'))
    ])
    render_shape = cv2.imread(
        osp.join(root_path, '/home/suyuze/workspace/ExAvatar_mine/data/XHumans/data/tvxq_mirotic/smplx_optimized/renders/0_smplx.jpg')
    ).shape[:2]

    def resize_like(img, target_size):
        return cv2.resize(img, (target_size[1], target_size[0]))

    for frame_idx in tqdm(frame_idx_list[0:5]):
        with open(osp.join(args.motion_path, 'cam_params', str(frame_idx) + '.json')) as f:
            cam_param = {k: torch.FloatTensor(v).cuda() for k,v in json.load(f).items()}
        with open(osp.join(args.motion_path, 'smplx_optimized', 'smplx_params_smoothed', str(frame_idx) + '.json')) as f:
            smplx_param = {k: torch.FloatTensor(v).cuda().view(-1) for k,v in json.load(f).items()}
        
        t = torch.tensor(frame_idx).cuda()/30
        with torch.no_grad():
            human_asset, human_asset_refined, human_offset, mesh_neutral_pose = tester.model.module.human_gaussian(
                smplx_param, cam_param, True, t
            )

            # ======== 简单编辑：放大鼻子区域 ========
            means = human_asset['mean_3d']
            scales = human_asset['scale']
            nose_center = means.mean(dim=0) + torch.tensor([0, 0.06, 0.02], device=means.device)
            dist = torch.norm(means - nose_center[None, :], dim=1)
            nose_ids = torch.where(dist < 0.03)[0]
            if len(nose_ids) > 0:
                means[nose_ids] += 0.5 * (means[nose_ids] - nose_center)
                scales[nose_ids] *= 1.3
                print(f"Edited {len(nose_ids)} nose Gaussians at frame {frame_idx}")
            human_asset['mean_3d'] = means
            human_asset['scale'] = scales
            # ========================================

            human_render = tester.model.module.gaussian_renderer(human_asset, render_shape, cam_param)

        # smplx mesh render（可保留对比）
        root_pose = smplx_param['root_pose'].view(1,3)
        body_pose = smplx_param['body_pose'].view(1,(len(smpl_x.joint_part['body'])-1)*3)
        jaw_pose = smplx_param['jaw_pose'].view(1,3)
        leye_pose = smplx_param['leye_pose'].view(1,3)
        reye_pose = smplx_param['reye_pose'].view(1,3)
        lhand_pose = smplx_param['lhand_pose'].view(1,len(smpl_x.joint_part['lhand'])*3)
        rhand_pose = smplx_param['rhand_pose'].view(1,len(smpl_x.joint_part['rhand'])*3)
        expr = smplx_param['expr'].view(1,smpl_x.expr_param_dim)
        trans = smplx_param['trans'].view(1,3)
        shape = tester.model.module.human_gaussian.shape_param[None]
        face_offset = smpl_x.face_offset.cuda()[None]
        joint_offset = tester.model.module.human_gaussian.joint_offset[None]
        output = tester.model.module.smplx_layer(
            global_orient=root_pose, body_pose=body_pose, jaw_pose=jaw_pose,
            leye_pose=leye_pose, reye_pose=reye_pose, left_hand_pose=lhand_pose,
            right_hand_pose=rhand_pose, expression=expr, betas=shape, transl=trans,
            face_offset=face_offset, joint_offset=joint_offset
        )
        mesh = output.vertices[0]
        mesh_render = render_mesh(mesh, smpl_x.face, cam_param, np.ones((render_shape[0],render_shape[1],3), dtype=np.float32)*255).astype(np.uint8)

        img0 = cv2.imread(osp.join(args.motion_path, 'frames/', f"{frame_idx:06d}.png"))
        img = cv2.imread(osp.join(args.motion_path, 'smplx_optimized/renders/', str(frame_idx) + '_smplx.jpg'))
        render = (human_render['img'].cpu().numpy().transpose(1,2,0)[:,:,::-1]*255).copy().astype(np.uint8)

        H, W = render.shape[:2]
        font_size = 1.5
        thick = 3
        cv2.putText(img0, 'Image', (int(1/3*img.shape[1]), int(0.05*img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, font_size, [51,51,255], thick, 2) 
        cv2.putText(img, 'Provided Mesh', (int(1/3*img.shape[1]), int(0.05*img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, font_size, [51,51,255], thick, 2) 
        cv2.putText(mesh_render, 'Predicted Mesh', (int(1/5*mesh_render.shape[1]), int(0.05*mesh_render.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, font_size, [51,51,255], thick, 2)
        cv2.putText(render, 'Edited Render', (int(1/3*render.shape[1]), int(0.05*render.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, font_size, [51,51,255], thick, 2) 

        img = resize_like(img, (H, W))
        mesh_render = resize_like(mesh_render, (H, W))
        img0 = resize_like(img0, (H, W))
        out = np.concatenate((img0, img, mesh_render, render), 1).astype(np.uint8)
        out = cv2.putText(out, str(frame_idx), (int(out.shape[1]*0.05), int(out.shape[0]*0.05)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, 2)

        # ==== 保存成图片 ====
        out_path = osp.join(args.output_dir, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(out_path, out)

if __name__ == "__main__":
    main()
