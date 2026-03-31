import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.layer import MeshRenderer
from nets.module import HumanGaussian, SMPLXParamDict, GaussianRenderer
from nets.loss import RGBLoss, SSIM, LPIPS, LaplacianReg, JointOffsetSymmetricReg, HandMeanReg,LabLoss,GradientLoss,BitPlaneLoss
from utils.flame import flame
from utils.smpl_x import smpl_x
import copy
from config import cfg
import cv2
import trimesh
from pytorch3d.structures import Meshes



from PIL import Image
import numpy as np
from scipy.spatial import KDTree

class Model(nn.Module):
    def __init__(self, human_gaussian, smplx_param_dict):
        super(Model, self).__init__()
        self.human_gaussian = human_gaussian
        self.smplx_param_dict = smplx_param_dict
        self.gaussian_renderer = GaussianRenderer()
        self.face_mesh_renderer = MeshRenderer(flame.vertex_uv, flame.face_uv)
        self.optimizable_params = self.human_gaussian.get_optimizable_params() 
        self.smplx_layer = copy.deepcopy(smpl_x.layer[cfg.smplx_gender])
        self.rgb_loss = RGBLoss()
        self.ssim = SSIM()
        self.lpips = LPIPS()
        self.lap_reg = LaplacianReg(smpl_x.vertex_num_upsampled, smpl_x.face_upsampled)
        self.joint_offset_sym_reg = JointOffsetSymmetricReg()
        self.hand_mean_reg = HandMeanReg()
        self.eval_modules = [self.lpips]
        self.labloss = LabLoss()
        self.gradient_loss = GradientLoss()
        self.bit_loss = BitPlaneLoss()
        # self.mp_pose = mp.solutions.pose
        # self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        

        # self.coco_regressor = np.load('/home/suyuze/workspace/ExAvatar_mine/data/J_regressor_coco.npy') 
    def get_smplx_outputs(self, smplx_param):
        
        
            root_pose = smplx_param['root_pose'].view(1, 3)
            body_pose = smplx_param['body_pose'].view(1, (len(smpl_x.joint_part['body']) - 1) * 3)
            jaw_pose = smplx_param['jaw_pose'].view(1, 3)
            leye_pose = smplx_param['leye_pose'].view(1, 3)
            reye_pose = smplx_param['reye_pose'].view(1, 3)
            lhand_pose = smplx_param['lhand_pose'].view(1, len(smpl_x.joint_part['lhand']) * 3)
            rhand_pose = smplx_param['rhand_pose'].view(1, len(smpl_x.joint_part['rhand']) * 3)
            expr = smplx_param['expr'].view(1, 50)
            trans = smplx_param['trans'].view(1, 3)
            
            shape = self.human_gaussian.shape_param[None]
            face_offset = smpl_x.face_offset.cuda()[None]
            joint_offset = self.human_gaussian.joint_offset[None]
            
            output = self.smplx_layer(
                global_orient=root_pose,
                body_pose=body_pose,
                jaw_pose=jaw_pose,
                leye_pose=leye_pose,
                reye_pose=reye_pose,
                left_hand_pose=lhand_pose,
                right_hand_pose=rhand_pose,
                expression=expr,
                betas=shape,
                transl=trans,
                face_offset=face_offset,
                joint_offset=joint_offset
            )
            
            mesh = output.vertices[0]
            return mesh


    def forward(self, data, mode):

        batch_size, _, img_height, img_width = data['img'].shape

        if mode == 'train':
            bg = torch.rand(3).float().cuda()  # 随机
        else:
            bg = torch.ones((3)).float().cuda()  # 白色

        human_assets, human_assets_refined, human_offsets, smplx_outputs = {}, {}, {}, []
        human_renders, human_renders_refined = {}, {}
        face_renders, face_renders_refined = [], []
        hand_renders, hand_renders_refined = {}, {}
        human_normald={}

        for i in range(batch_size):
            
            
            time_ = data['frame_idx']/30
            # 从 human Gaussians 获取资产和偏移
            smplx_param = self.smplx_param_dict([data['capture_id'][i]], [data['frame_idx'][i]])[0]
            human_asset, human_asset_refined, human_offset, mesh_neutral_pose = self.human_gaussian(smplx_param,data['cam_param'],True,time_)

            #test的时候可以放出来
            # ply_path = "/home/suyuze/workspace/ExAvatar_mine/0724best/result/00028/test/Take14/00095_point.ply"
            # a = trimesh.load(ply_path)
            # human_asset_refined['mean_3d'] = torch.tensor(a.vertices).float().cuda()

            
            # ======== 简单编辑：放大鼻子区域 ========
            # means = human_asset['mean_3d']
            # scales = human_asset['scale']

            # # 以平均位置略微上移作为鼻子中心
            # nose_center = means.mean(dim=0) + torch.tensor([0, 0.08, 0.04], device=means.device)

            # # 找出靠近鼻子中心的点（扩大范围）
            # dist = torch.norm(means - nose_center[None, :], dim=1)
            # nose_ids = torch.where(dist < 0.05)[0]


            # ========================================

            key_list = ['mean_3d', 'scale', 'rotation', 'rgb']
            if (mode == 'train') and cfg.is_warmup:
                human_asset['scale_wo_clamp'] = human_asset['scale'].clone()
                human_asset['scale'] = torch.clamp(human_asset['scale'], max=0.001)
                human_asset_refined['scale_wo_clamp'] = human_asset_refined['scale'].clone()
                human_asset_refined['scale'] = torch.clamp(human_asset_refined['scale'], max=0.001)
                key_list += ['scale_wo_clamp']

            for key in key_list:
                if key not in human_assets:
                    human_assets[key] = [human_asset[key]]
                    human_assets_refined[key] = [human_asset_refined[key]]
                else:
                    human_assets[key].append(human_asset[key])
                    human_assets_refined[key].append(human_asset_refined[key])

            for key in ['mean_offset', 'mean_offset_offset', 'scale_offset', 'rgb_offset']:
                if key not in human_offsets:
                    human_offsets[key] = [human_offset[key]]
                else:
                    human_offsets[key].append(human_offset[key])

            smplx_output = self.get_smplx_outputs(smplx_param)
            smplx_outputs.append(smplx_output)
            lhand_tensor = torch.tensor(smpl_x.lhand_vertex_idx)
            rhand_tensor = torch.tensor(smpl_x.rhand_vertex_idx)
            all_hand_vertex_idx = torch.cat([lhand_tensor, rhand_tensor], dim=0)

            gt_hpoints = smplx_output[all_hand_vertex_idx]
                              
            human_render = self.gaussian_renderer(human_asset, (img_height, img_width), {k: v[i] for k,v in data['cam_param'].items()}, bg)
            for key in ['img', 'mask']:
                if key not in human_renders:
                    human_renders[key] = [human_render[key]]
                else:
                    human_renders[key].append(human_render[key])
            
            with torch.no_grad():
                normal = Meshes(verts=human_asset_refined['mean_3d'][None,:,:],
                                faces=torch.LongTensor(smpl_x.face_upsampled).cuda()[None,:,:]).verts_normals_packed().reshape(1,smpl_x.vertex_num_upsampled,3).detach().repeat(batch_size,1,1)
            normals_vis = (normal + 1.0) / 2.0
            human_render_refined = self.gaussian_renderer(human_asset_refined, (img_height, img_width), {k: v[i] for k,v in data['cam_param'].items()}, bg)
            human_asset_refined['rgb'] = normals_vis
            human_normal = self.gaussian_renderer(human_asset_refined, (img_height, img_width), {k: v[i] for k,v in data['cam_param'].items()}, bg)
            for key in ['img', 'mask']:
                if key not in human_normald:
                    human_normald[key] = [human_normal[key]]
                else:
                    human_normald[key].append(human_normal[key])
            for key in ['img', 'mask']:
                if key not in human_renders_refined:
                    human_renders_refined[key] = [human_render_refined[key]]
                else:
                    human_renders_refined[key].append(human_render_refined[key])

            # face render
            face_texture, face_texture_mask = flame.texture[None], flame.texture_mask[None,0:1]
            face_texture = torch.cat((face_texture, face_texture_mask),1)
            face_render = self.face_mesh_renderer(face_texture, human_asset['mean_3d'][None,smpl_x.face_vertex_idx,:], flame.face, {k: v[i,None] for k,v in data['cam_param'].items()}, (img_height, img_width)) 
            face_render_refined = self.face_mesh_renderer(face_texture, human_asset_refined['mean_3d'][None,smpl_x.face_vertex_idx,:], flame.face, {k: v[i,None] for k,v in data['cam_param'].items()}, (img_height, img_width)) 
            face_renders.append(face_render[0])
            face_renders_refined.append(face_render_refined[0])
            
            
            
            # # '''对手部区域特殊处理
            # is_hand = (self.human_gaussian.is_lhand + self.human_gaussian.is_rhand) > 0  # shape: [N], bool
            # is_hand = is_hand.bool()
            # print(is_hand.shape, is_hand.dtype, is_hand.device)
            # hand_gaussians1 = {
            #     'mean_3d': human_asset['mean_3d'][is_hand],        # [N_hand, 3]
            #     'opacity': human_asset['opacity'][is_hand],        # [N_hand, 1]
            #     'scale': human_asset['scale'][is_hand],            # [N_hand, 3]
            #     'rotation': human_asset['rotation'][is_hand],      # [N_hand, 4 or 6]
            #     'rgb': human_asset['rgb'][is_hand],                # [N_hand, 3]
            # }
            # hand_gaussians2 = {
            #     'mean_3d': human_asset_refined['mean_3d'][is_hand],        # [N_hand, 3]
            #     'opacity': human_asset_refined['opacity'][is_hand],        # [N_hand, 1]
            #     'scale': human_asset_refined['scale'][is_hand],            # [N_hand, 3]
            #     'rotation': human_asset_refined['rotation'][is_hand],      # [N_hand, 4 or 6]
            #     'rgb': human_asset_refined['rgb'][is_hand],                # [N_hand, 3]
            # }
            # hand_renders1 = self.gaussian_renderer(
            #     hand_gaussians1, (img_height, img_width), {k: v[i] for k, v in data['cam_param'].items()}, bg
            # )
            # hand_renders2 = self.gaussian_renderer(
            #     hand_gaussians2, (img_height, img_width), {k: v[i] for k, v in data['cam_param'].items()}, bg
            # )

            
            # for key in ['img', 'mask']:
            #     if key not in hand_renders:
            #         hand_renders[key] = [hand_renders1[key]]
            #     else:
            #         hand_renders[key].append(hand_renders1[key])
            # for key in ['img', 'mask']:
            #     if key not in hand_renders_refined:
            #         hand_renders_refined[key] = [hand_renders2[key]]
            #     else:
            #         hand_renders_refined[key].append(hand_renders2[key])

            # # '''
        human_assets = {k: torch.stack(v) for k, v in human_assets.items()}
        human_assets_refined = {k: torch.stack(v) for k, v in human_assets_refined.items()}
        human_offsets = {k: torch.stack(v) for k, v in human_offsets.items()}
        smplx_outputs = torch.stack(smplx_outputs)
        human_renders = {k: torch.stack(v) for k, v in human_renders.items()}
        human_renders_refined = {k: torch.stack(v) for k, v in human_renders_refined.items()}
        face_renders = torch.stack(face_renders)
        face_renders_refined = torch.stack(face_renders_refined)
        
        hand_renders = {k: torch.stack(v) for k, v in hand_renders.items()}
        hand_renders_refined = {k: torch.stack(v) for k, v in hand_renders_refined.items()}
        human_normald = {k: torch.stack(v) for k, v in human_normald.items()}



        if mode == 'train':

            
            
            
            # 计算损失
            loss = {}
            
            # loss['bit'] = self.bit_loss(human_renders['img'], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None]) * 0.8
            # loss['bit_refined'] = self.bit_loss(human_renders_refined['img'], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None]) * 0.8
            loss['lab'] = self.labloss(human_renders['img'], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None]) * 0.3
            loss['lab_refined'] = self.labloss(human_renders_refined['img'], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None]) * 0.3
            
            
            loss['gradient'] = self.gradient_loss(human_renders['img'], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None]) * 0.01
            loss['gradient_refined'] = self.gradient_loss(human_renders_refined['img'], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None]) * 0.01
            loss['rgb'] = self.rgb_loss(human_renders['img'], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None]) * cfg.rgb_loss_weight
            loss['ssim'] = (1 - self.ssim(human_renders['img'], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None])) * cfg.ssim_loss_weight
            loss['lpips'] = self.lpips(human_renders['img'], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None]) * cfg.lpips_weight
            is_face = ((face_renders[:,:3] != -1) * (face_renders[:,3:] == 1)).float()
            loss['rgb_face'] = self.rgb_loss(human_renders['img'] * (1 - is_face) + face_renders[:,:3] * is_face, data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None]) * cfg.rgb_loss_weight

            loss['rgb_refined'] = self.rgb_loss(human_renders_refined['img'], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None]) * cfg.rgb_loss_weight
            loss['ssim_refined'] = (1 - self.ssim(human_renders_refined['img'], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None])) * cfg.ssim_loss_weight
            loss['lpips_refined'] = self.lpips(human_renders_refined['img'], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None]) * cfg.lpips_weight
            is_face = ((face_renders_refined[:,:3] != -1) * (face_renders_refined[:,3:] == 1)).float()
            loss['rgb_face_refined'] = self.rgb_loss(human_renders_refined['img'] * (1 - is_face) + face_renders_refined[:,:3] * is_face, data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None]) * cfg.rgb_loss_weight

            weight = torch.ones((1,smpl_x.vertex_num_upsampled,1)).float().cuda() * 10
            weight[:,self.human_gaussian.is_rhand,:] = 1000
            weight[:,self.human_gaussian.is_lhand,:] = 1000
            weight[:,self.human_gaussian.is_face,:] = 1
            weight[:,self.human_gaussian.is_face_expr,:] = 10
            loss['gaussian_mean_reg'] = (human_offsets['mean_offset'] ** 2 + human_offsets['mean_offset_offset'] ** 2) * weight
            loss['gaussian_mean_hand_reg'] = self.hand_mean_reg(mesh_neutral_pose, human_offsets['mean_offset'], self.human_gaussian.is_lhand, self.human_gaussian.is_rhand) + self.hand_mean_reg(mesh_neutral_pose, human_offsets['mean_offset_offset'], self.human_gaussian.is_lhand, self.human_gaussian.is_rhand)
            weight = torch.ones((1,smpl_x.vertex_num_upsampled,1)).float().cuda()
            weight[:,self.human_gaussian.is_rhand,:] = 1000 
            weight[:,self.human_gaussian.is_lhand,:] = 1000
            weight[:,self.human_gaussian.is_face_expr,:] = 10
            weight[:,self.human_gaussian.is_cavity,:] = 0
            if cfg.is_warmup:
                loss['gaussian_scale_reg'] = (human_assets['scale_wo_clamp'] ** 2 + human_offsets['scale_offset'] ** 2) * weight
            else:
                loss['gaussian_scale_reg'] = (human_assets['scale'] ** 2 + human_offsets['scale_offset'] ** 2) * weight
            
            weight = torch.ones((1,smpl_x.vertex_num_upsampled,1)).float().cuda()
            weight[:,self.human_gaussian.is_face_expr,:] = 50
            weight[:,self.human_gaussian.is_cavity,:] = 0.1
            loss['lap_mean'] = (self.lap_reg(mesh_neutral_pose[None,:,:].detach() + human_offsets['mean_offset'], mesh_neutral_pose[None,:,:].detach()) + \
                                self.lap_reg(mesh_neutral_pose[None,:,:].detach() + human_offsets['mean_offset'] + human_offsets['mean_offset_offset'], mesh_neutral_pose[None,:,:].detach())) * 100000 * weight
            weight = torch.ones((1,smpl_x.vertex_num_upsampled,1)).float().cuda()
            weight[:,self.human_gaussian.is_cavity,:] = 0.1
            loss['lap_scale'] = (self.lap_reg(human_assets['scale'], None) + self.lap_reg(human_assets_refined['scale'], None)) * 100000 * weight
          
            weight = torch.ones((smpl_x.joint_num,3)).float().cuda()
            weight[smpl_x.joint_part['lhand'],:] = 10
            weight[smpl_x.joint_part['rhand'],:] = 10
            loss['joint_offset_reg'] = (self.human_gaussian.joint_offset - smpl_x.joint_offset.cuda()) ** 2 * weight
            loss['joint_offset_sym_reg'] = self.joint_offset_sym_reg(self.human_gaussian.joint_offset)
            return loss
        else:
            out = {}


            out['human_img'] = human_renders['img']
            out['human_img_refined'] = human_renders_refined['img']
            out['smplx_mesh'] = smplx_outputs

            is_face = (face_renders[:,:3] != -1).float() * face_renders[:,3:]
            out['human_face_img'] = human_renders['img'] * (1 - is_face) + face_renders[:,:3] * is_face
            is_face = (face_renders_refined[:,:3] != -1).float() * face_renders_refined[:,3:]
            out['human_face_img_refined'] = human_renders_refined['img'] * (1 - is_face) + face_renders_refined[:,:3] * is_face
            # out['hand'] = hand_renders_refined['img']
            # out['hand_mask'] = hand_renders_refined['mask']
            # out['hand_gaussian'] = hand_gaussians2
            # out['gt_hpoints'] = gt_hpoints
        
            # out['mesh'] = mesh # [N, 3]
            out['human_asset_refined'] = human_asset_refined['mean_3d']
            out['human_normald'] = human_normald['img']
            return out

           

def get_model(smplx_params):

    human_gaussian = HumanGaussian()
    with torch.no_grad():
        human_gaussian.init()

    if smplx_params is not None:
        smplx_param_dict = SMPLXParamDict()
        with torch.no_grad():
            smplx_param_dict.init(smplx_params)
    else:
        smplx_param_dict = None

    model = Model(human_gaussian, smplx_param_dict)
    return model

