import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_matrix, axis_angle_to_matrix, matrix_to_axis_angle
from pytorch3d.ops import knn_points
from utils.transforms import eval_sh, RGB2SH, get_fov, get_view_matrix, get_proj_matrix
from utils.smpl_x import smpl_x
from smplx.lbs import batch_rigid_transform
# from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization_depth import GaussianRasterizationSettings, GaussianRasterizer
from nets.layer import make_linear_layers
from pytorch3d.structures import Meshes
from config import cfg
import copy
import numpy as np
from smplx import create
import os.path as osp
# import trimesh
import pickle
import cv2

import os
import torch
import torch.nn as nn

class StructureAwareOffsetPredictor(nn.Module):
    def __init__(self, 
                 tri_feat_dim=96,
                 joint_embed_dim=6+3+16,
                 hidden_dim=128,
                 num_joints=55):
        super().__init__()

        self.joint_id_embed = nn.Embedding(num_joints, 16)

        self.fusion_net = nn.Sequential(
            nn.Linear(tri_feat_dim + joint_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 依然输出3，但含义变为 (normal_weight, tangent1_weight, tangent2_weight)
        self.manifold_offset_net = nn.Linear(hidden_dim, 3) 
        self.Tscale_offset_net = nn.Linear(hidden_dim, 1)

    def get_tangent_basis(self, normals):
        """
        根据法向量构造局部正交基 (Gram-Schmidt 简版)
        normals: [N, 3]
        """
        # 选择一个不与法线平行的向量作为参考
        ref = torch.tensor([1.0, 0.0, 0.0], device=normals.device).expand_as(normals)
        # 如果法线恰好是 x 轴，换一个参考向量
        is_x_axis = torch.abs(normals[:, 0]) > 0.9
        ref[is_x_axis] = torch.tensor([0.0, 1.0, 0.0], device=normals.device)

        # 计算两个切向量
        tan1 = torch.cross(normals, ref, dim=-1)
        tan1 = nn.functional.normalize(tan1, dim=-1)
        tan2 = torch.cross(normals, tan1, dim=-1)
        tan2 = nn.functional.normalize(tan2, dim=-1)
        
        return tan1, tan2

    def forward(self, tri_feat, bound_joint_ids, joint_rot6d, joint_pos, normals):
        """
        新增参数: normals [N, 3] (对应 Canonical space 的法线)
        """
        N = tri_feat.shape[0]
        device = tri_feat.device

        # 1. 构造 joint 特征
        J = joint_rot6d.shape[0]
        joint_id_embed = self.joint_id_embed(torch.arange(J, device=device))
        joint_feat_all = torch.cat([joint_rot6d, joint_pos, joint_id_embed], dim=-1)

        # 2. 融合特征
        selected_joint_feat = joint_feat_all[bound_joint_ids]
        fused_feat = torch.cat([tri_feat, selected_joint_feat], dim=-1)
        feat = self.fusion_net(fused_feat)

        # 3. 预测流形系数
        # weights: [N, 3] -> (w_n, w_t1, w_t2)
        weights = self.manifold_offset_net(feat)
        
        # 4. 构造局部坐标系并映射回世界坐标偏移
        # 这样预测的 offset 会严格参考表面法线
        tan1, tan2 = self.get_tangent_basis(normals)
        
        # 核心：将标量系数转回 3D 偏移向量
        mean_offset = (weights[:, :1] * normals + 
                       weights[:, 1:2] * tan1 + 
                       weights[:, 2:3] * tan2)
        
        scale_offset = self.Tscale_offset_net(feat)
        
        return mean_offset, scale_offset
#___________________
# forward_geo_network
'''
class StructureAwareOffsetPredictor(nn.Module):
    def __init__(self, 
                 tri_feat_dim=96,
                 joint_embed_dim=6+3+16,
                 hidden_dim=128,
                 num_joints=55):
        super().__init__()

        self.joint_id_embed = nn.Embedding(num_joints, 16)

        self.fusion_net = nn.Sequential(
            nn.Linear(tri_feat_dim + joint_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.Tmean_offset_net = nn.Linear(hidden_dim, 3)
        self.Tscale_offset_net = nn.Linear(hidden_dim, 1)

    def forward(self, tri_feat, bound_joint_ids, joint_rot6d, joint_pos):
        """
        tri_feat: [N, C]
        bound_joint_ids: [N]
        joint_rot6d: [J, 6]
        joint_pos: [J, 3]
        """
        N = tri_feat.shape[0]
        device = tri_feat.device

        # 构造 joint 特征
        J = joint_rot6d.shape[0]
        joint_id_embed = self.joint_id_embed(torch.arange(J, device=device))  # [J, 16]
        joint_feat_all = torch.cat([joint_rot6d, joint_pos, joint_id_embed], dim=-1)  # [J, 25]

        # 对每个点，选出对应的 joint 特征
        selected_joint_feat = joint_feat_all[bound_joint_ids]  # [N, 25]

        # 融合后预测
        fused_feat = torch.cat([tri_feat, selected_joint_feat], dim=-1)  # [N, 128+25]
        feat = self.fusion_net(fused_feat)

        mean_offset_offset = self.Tmean_offset_net(feat)
        scale_offset = self.Tscale_offset_net(feat)
        print( "scale_offset:", scale_offset.shape)
        return mean_offset_offset, scale_offset

'''


class HumanGaussian(nn.Module):
    def __init__(self):
        super(HumanGaussian, self).__init__()
        # 初始化一个三维平面参数，用于存储模型的几何信息，使用nn.Parameter使其成为可学习的参数
        # .float()确保数据类型为浮点型，.cuda()将数据移至GPU上
        self.triplane = nn.Parameter(torch.zeros((3,*cfg.triplane_shape)).float().cuda())
        ## 同上，但这个三维平面参数专门用于人脸部分
        self.triplane_face = nn.Parameter(torch.zeros((3,*cfg.triplane_shape)).float().cuda())
        
        self.mlp_mano = nn.Sequential(
            nn.Linear(3 + 90, 64),  # 3 for offset, 30 for condition
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # refined offset
        )
        
        
        # 六个平面（xy, xz, yz, xt, yt, zt）
        self.hexplane = nn.Parameter(torch.zeros((6, cfg.hexplane_n_channels, *cfg.hexplane_resolution)).float())
        self.hexplane_face = nn.Parameter(torch.zeros((6, cfg.hexplane_n_channels, *cfg.hexplane_resolution)).float())
        #triplane_shape[0] 就是cfg.hexplane_n_channels
        
        self.hex_to_tri = nn.Linear(cfg.triplane_shape[0]*6, cfg.triplane_shape[0]*3) 
        
        # 创建一个几何网络，用于处理和转换几何信息
        # make_linear_layers函数创建一系列全连接层，列表中的数字表示各层的神经元数量
        # use_gn=True表示使用Group Normalization
        self.geo_net = make_linear_layers([cfg.triplane_shape[0]*3, 128, 128, 128], use_gn=True)
           # 创建一个网络，用于计算均值偏移量
         # relu_final=False表示最后一层不使用ReLU激活函数    
        self.mean_offset_net = make_linear_layers([128, 3], relu_final=False)
        # 创建一个网络，用于计算缩放因子
        self.scale_net = make_linear_layers([128, 1], relu_final=False)
        # 创建另一个几何偏移网络，输入维度包括了额外的关节信息
        
        
        # self.geo_offset_net = make_linear_layers([cfg.triplane_shape[0]*3+(smpl_x.joint_num-1)*6, 128, 128, 128], use_gn=True)
        # # 创建一个网络，用于计算均值偏移量的偏移量
        # self.mean_offset_offset_net = make_linear_layers([128, 3], relu_final=False)
        # # 创建一个网络，用于计算缩放因子的偏移量
        # self.scale_offset_net = make_linear_layers([128, 1], relu_final=False)
        # 创建一个RGB网络，用于生成颜色信息
        # relu_final=False表示最后一层不使用ReLU激活函数，use_gn=True表示使用Group Normalization
        self.rgb_net = make_linear_layers([cfg.triplane_shape[0]*3, 128, 128, 128, 3], relu_final=False, use_gn=True)
        # 创建一个RGB偏移网络，输入维度包括了额外的关节信息和颜色信息cfg.triplane_shape[0]*3+(smpl_x.joint_num-1)*6+3
        self.rgb_offset_net = make_linear_layers([105, 128, 128, 128, 3], relu_final=False, use_gn=True)
        # 深度复制SMPL-X层结构，并将其移至GPU上
        self.smplx_layer = copy.deepcopy(smpl_x.layer[cfg.smplx_gender]).cuda()#只选择了男性
        # 初始化形状参数，使其成为可学习的参数
        self.shape_param = nn.Parameter(smpl_x.shape_param.float().cuda())
        # 初始化关节偏移参数，使其成为可学习的参数
        self.joint_offset = nn.Parameter(smpl_x.joint_offset.float().cuda())
        
        self.alpha_mlp = nn.Sequential(
            nn.Linear(cfg.triplane_shape[0]*3* 2, cfg.triplane_shape[0]*3),
            nn.ReLU(),
            nn.Linear(cfg.triplane_shape[0]*3, cfg.triplane_shape[0]*3),
            nn.Sigmoid()  # 输出每个点的注意力权重
        )
        
        

        # 初始化 MANO 模型
        self.mano_right_layer = create(
            model_path='/home/suyuze/workspace/ExAvatar_mine/common/utils/human_model_files/smplx/MANO_RIGHT.pkl',   # 路径下应包含 mano/MANO_RIGHT.pkl
            model_type='mano',
            is_rhand=True,
            use_pca=False
        )

        self.mano_left_layer = create(
            model_path='/home/suyuze/workspace/ExAvatar_mine/common/utils/human_model_files/smplx/MANO_LEFT.pkl',   # 路径下应包含 mano/MANO_LEFT.pkl
            model_type='mano',
            is_rhand=False,
            use_pca=False
        )
        # self.forward_geo_network = StructureAwareOffsetPredictor(
        #     tri_feat_dim=96,
        #     joint_embed_dim=6+3+16,  # rot6d + joint_pos + joint_id_emb
        #     # d_model=64,  1有
        #     hidden_dim=128,
        #     num_joints=55
        # )
        
        self.forward_geo_network =StructureAwareOffsetPredictor(
            tri_feat_dim=96,
            joint_embed_dim=6+3+16,  # rot6d + joint_pos + joint_id_emb
            # d_model=64,  1有
            hidden_dim=128,
            num_joints=55
        )

        



        


        
     
    def init(self):
        # upsample mesh and other assets # 上采样网格和其他资产
        xyz, _, _, _ = self.get_neutral_pose_human(jaw_zero_pose=False, use_id_info=False)
        # 获取皮肤权重
        skinning_weight = self.smplx_layer.lbs_weights.float()
        # 调整姿态方向的维度顺序并重塑 smpl_x.joint_num：模型的关节数，表示模型有多少个关节。
        pose_dirs = self.smplx_layer.posedirs.permute(1,0).reshape(smpl_x.vertex_num,3*(smpl_x.joint_num-1)*9)
        # 重塑表情方向
        expr_dirs = self.smplx_layer.expr_dirs.view(smpl_x.vertex_num,3*smpl_x.expr_param_dim)
        # 初始化手部、面部和表情的指示器 都一样
        is_rhand, is_lhand, is_face, is_face_expr = torch.zeros((smpl_x.vertex_num,1)).float().cuda(), torch.zeros((smpl_x.vertex_num,1)).float().cuda(), torch.zeros((smpl_x.vertex_num,1)).float().cuda(), torch.zeros((smpl_x.vertex_num,1)).float().cuda()
        # 标记特定顶点为手部、面部或表情
        is_rhand[smpl_x.rhand_vertex_idx], is_lhand[smpl_x.lhand_vertex_idx], is_face[smpl_x.face_vertex_idx], is_face_expr[smpl_x.expr_vertex_idx] = 1.0, 1.0, 1.0, 1.0
        # 获取腔体的指示器
        is_cavity = torch.FloatTensor(smpl_x.is_cavity).cuda()[:,None]
        # 使用虚拟顶点上采样网格 增加网格分辨率的技术
        _, skinning_weight, pose_dirs, expr_dirs, is_rhand, is_lhand, is_face, is_face_expr, is_cavity = smpl_x.upsample_mesh(torch.ones((smpl_x.vertex_num,3)).float().cuda(), [skinning_weight, pose_dirs, expr_dirs, is_rhand, is_lhand, is_face, is_face_expr, is_cavity]) # upsample with dummy vertex
        # 重塑姿态方向
        pose_dirs = pose_dirs.reshape(smpl_x.vertex_num_upsampled*3,(smpl_x.joint_num-1)*9).permute(1,0) 
        # 重塑表情方向
        expr_dirs = expr_dirs.view(smpl_x.vertex_num_upsampled,3,smpl_x.expr_param_dim)
        # 将指示器转换为布尔值
        is_rhand, is_lhand, is_face, is_face_expr = is_rhand[:,0] > 0, is_lhand[:,0] > 0, is_face[:,0] > 0, is_face_expr[:,0] > 0
        
        is_cavity = is_cavity[:,0] > 0
        # 注册缓冲区，这些变量不会被优化器更新，但会保存到模型的状态字典中
        self.register_buffer('pos_enc_mesh', xyz)
        self.register_buffer('skinning_weight', skinning_weight)
        self.register_buffer('pose_dirs', pose_dirs)
        self.register_buffer('expr_dirs', expr_dirs)
        self.register_buffer('is_rhand', is_rhand)
        self.register_buffer('is_lhand', is_lhand)
        self.register_buffer('is_face', is_face)
        self.register_buffer('is_face_expr', is_face_expr)
        self.register_buffer('is_cavity', is_cavity)

    def get_optimizable_params(self):
        # 定义可优化参数列表
        optimizable_params = [
             # 三维平面参数，用于存储人体模型的几何信息
            {'params': [self.triplane], 'name': 'triplane_human', 'lr': cfg.lr},
            # 三维平面参数，专门用于人脸部分
            {'params': [self.triplane_face], 'name': 'triplane_face_human', 'lr': cfg.lr}, 
            {'params': [self.hexplane], 'name': 'hexplane_human', 'lr': cfg.lr},
            # 三维平面参数，专门用于人脸部分
            {'params': [self.hexplane_face], 'name': 'hexplane_face_human', 'lr': cfg.lr},
            
            {'params': list(self.hex_to_tri.parameters()), 'name': 'hex_to_tri_human', 'lr': cfg.lr},
            {'params': list(self.alpha_mlp.parameters()), 'name': 'alpha_mlp_human', 'lr': cfg.lr},
            {'params': list(self.mlp_mano.parameters()), 'name': 'mlp_mano_human', 'lr': cfg.lr},
            {'params': list(self.forward_geo_network.parameters()), 'name': 'forward_geo_network_human', 'lr': cfg.lr},
            
            # # 几何网络的参数
            {'params': list(self.geo_net.parameters()), 'name': 'geo_net_human', 'lr': cfg.lr},
            # # 均值偏移网络的参数
            {'params': list(self.mean_offset_net.parameters()), 'name': 'mean_offset_net_human', 'lr': cfg.lr},
            # # 缩放网络的参数
            {'params': list(self.scale_net.parameters()), 'name': 'scale_net_human', 'lr': cfg.lr},
            # # 几何偏移网络的参数
            # {'params': list(self.geo_offset_net.parameters()), 'name': 'geo_offset_net_human', 'lr': cfg.lr},
            
            #均值偏移偏移网络的参数
            # {'params': list(self.mean_offset_offset_net.parameters()), 'name': 'mean_offset_offset_net_human', 'lr': cfg.lr},
            #缩放偏移网络的参数
            # {'params': list(self.scale_offset_net.parameters()), 'name': 'scale_offset_net_human', 'lr': cfg.lr},
            # RGB网络的参数
            {'params': list(self.rgb_net.parameters()), 'name': 'rgb_net_human', 'lr': cfg.lr},
            # RGB偏移网络的参数
            {'params': list(self.rgb_offset_net.parameters()), 'name': 'rgb_offset_net_human', 'lr': cfg.lr},
            # 形状参数
            {'params': [self.shape_param], 'name': 'shape_param_human', 'lr': cfg.lr},
            # 关节偏移参数
            {'params': [self.joint_offset], 'name': 'joint_offset_human', 'lr': cfg.lr}
        ]
        return optimizable_params
    
    #生成中立姿态下的人体模型，并计算将大姿态转换为零姿态的变换矩阵
    def get_neutral_pose_human(self, jaw_zero_pose, use_id_info):
        # 定义中立姿态下的零姿态
        zero_pose = torch.zeros((1, 3)).float().cuda()
        # 获取中立姿态下的身体姿态
        neutral_body_pose = smpl_x.neutral_body_pose.view(1, -1).cuda()
        # 定义中立姿态下的手部姿态为零
        zero_hand_pose = torch.zeros((1, len(smpl_x.joint_part['lhand']) * 3)).float().cuda()
        # 定义中立姿态下的表情参数为零
        zero_expr = torch.zeros((1, smpl_x.expr_param_dim)).float().cuda()
        
        # 根据jaw_zero_pose参数决定是否使用闭合嘴巴的姿态
        if jaw_zero_pose:
            jaw_pose = torch.zeros((1, 3)).float().cuda()
        else:
            jaw_pose = smpl_x.neutral_jaw_pose.view(1, 3).cuda()  # 打开嘴巴的姿态
        
        # 根据use_id_info参数决定是否使用身份信息
        if use_id_info:
            shape_param = self.shape_param[None, :]  # 使用模型的形状参数
            face_offset = smpl_x.face_offset[None, :, :].float().cuda()  # 使用模型的脸部偏移
            joint_offset = smpl_x.get_joint_offset(self.joint_offset[None, :, :])  # 获取关节偏移
        else:
            shape_param = torch.zeros((1, smpl_x.shape_param_dim)).float().cuda()  # 不使用身份信息时，形状参数为零
            face_offset = None
            joint_offset = None
        
        # 使用SMPL-X层生成中立姿态下的人体模型
        output = self.smplx_layer(
            global_orient=zero_pose,
            body_pose=neutral_body_pose,
            left_hand_pose=zero_hand_pose,
            right_hand_pose=zero_hand_pose,
            jaw_pose=jaw_pose,
            leye_pose=zero_pose,
            reye_pose=zero_pose,
            expression=zero_expr,
            betas=shape_param,
            face_offset=face_offset,
            joint_offset=joint_offset
        )
        
        # 获取中立姿态下的网格顶点
        mesh_neutral_pose = output.vertices[0]
        # 对网格进行上采样
        mesh_neutral_pose_upsampled = smpl_x.upsample_mesh(mesh_neutral_pose)
        # 获取中立姿态下的关节位置
        joint_neutral_pose = output.joints[0][:smpl_x.joint_num, :]
        
        # 计算将大姿态转换为零姿态的变换矩阵
        neutral_body_pose = neutral_body_pose.view(len(smpl_x.joint_part['body']) - 1, 3)
        zero_hand_pose = zero_hand_pose.view(len(smpl_x.joint_part['lhand']), 3)
        neutral_body_pose_inv = matrix_to_axis_angle(torch.inverse(axis_angle_to_matrix(neutral_body_pose)))
        jaw_pose_inv = matrix_to_axis_angle(torch.inverse(axis_angle_to_matrix(jaw_pose)))
        pose = torch.cat((zero_pose, neutral_body_pose_inv, jaw_pose_inv, zero_pose, zero_pose, zero_hand_pose, zero_hand_pose))
        pose = axis_angle_to_matrix(pose)
        _, transform_mat_neutral_pose = batch_rigid_transform(pose[None, :, :, :], joint_neutral_pose[None, :, :], self.smplx_layer.parents)
        transform_mat_neutral_pose = transform_mat_neutral_pose[0]
        
        
         # === 添加保存.obj逻辑 ===

        # mesh_np = mesh_neutral_pose_upsampled.detach().cpu().numpy()
        # faces = smpl_x.face_upsampled  # 假设 smpl_x 里有 faces 属性，shape 为 [F, 3]
        # out_path = '/home/suyuze/workspace/ExAvatar_mine/1.obj'  # 你可以改路径
        # neutral_mesh = trimesh.Trimesh(vertices=mesh_np, faces=faces, process=False)
        # neutral_mesh.export(out_path)
        # print(f"[INFO] Neutral pose mesh saved to {out_path}")
        
        
        # 返回上采样后的网格、原始网格、关节位置和变换矩阵
        return mesh_neutral_pose_upsampled, mesh_neutral_pose, joint_neutral_pose, transform_mat_neutral_pose



    #生成一个零姿态的人体模型，即所有关节都处于自然状态，没有任何旋转或位移
    def get_zero_pose_human(self, return_mesh=False):
        # 创建零姿态的各个部分的张量
        zero_pose = torch.zeros((1,3)).float().cuda()  # 全局姿态
        ''' SMPL-X 模型中的每个关节都有三个姿态参数（对应于三维空间中的旋转），除了根关节（global orientation）之外。
        len(smpl_x.joint_part['body']) 是身体部分的关节数量，减去 1 是为了排除根关节'''
        zero_body_pose = torch.zeros((1,(len(smpl_x.joint_part['body'])-1)*3)).float().cuda()  # 身体姿态
        zero_hand_pose = torch.zeros((1,len(smpl_x.joint_part['lhand'])*3)).float().cuda()  # 左手姿态
        zero_expr = torch.zeros((1,smpl_x.expr_param_dim)).float().cuda()  # 表情参数
        
        # 获取形状参数、面部偏移和关节偏移
        shape_param = self.shape_param[None,:]
        face_offset = smpl_x.face_offset[None,:,:].float().cuda()
        joint_offset = smpl_x.get_joint_offset(self.joint_offset[None,:,:])
        
        # 使用 SMPL-X 模型生成零姿态的人体模型
        output = self.smplx_layer(global_orient=zero_pose, body_pose=zero_body_pose, left_hand_pose=zero_hand_pose, 
                                  right_hand_pose=zero_hand_pose, jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose, 
                                  expression=zero_expr, betas=shape_param, face_offset=face_offset, joint_offset=joint_offset)
        
        # 提取关节位置
        joint_zero_pose = output.joints[0][:smpl_x.joint_num,:]
        
        # 如果不需要返回网格模型，则直接返回关节位置
        if not return_mesh:
            return joint_zero_pose
        else:
            # 否则，提取网格顶点并进行上采样
            mesh_zero_pose = output.vertices[0]
            mesh_zero_pose_upsampled = smpl_x.upsample_mesh(mesh_zero_pose)
            # 返回上采样后的网格模型、原始网格模型和关节位置
            return mesh_zero_pose_upsampled, mesh_zero_pose, joint_zero_pose
    # 计算从一个大姿态（例如，一个人站立的姿态）到一个图像姿态（例如，一个人在照片中的姿态）的转换矩阵 实现不同姿态之间的转换
    def get_transform_mat_joint(self, transform_mat_neutral_pose, joint_zero_pose, smplx_param):
        # 1. 大 pose -> zero pose# 使用输入的 transform_mat_neutral_pose 作为从大姿态到零姿态的转换矩阵
        transform_mat_joint_1 = transform_mat_neutral_pose

        # 2. zero pose -> image pose# 将 smplx_param 中的各种姿态参数转换为正确的形状
        root_pose = smplx_param['root_pose'].view(1,3)
        body_pose = smplx_param['body_pose'].view(len(smpl_x.joint_part['body'])-1,3)
        jaw_pose = smplx_param['jaw_pose'].view(1,3)
        leye_pose = smplx_param['leye_pose'].view(1,3)
        reye_pose = smplx_param['reye_pose'].view(1,3)
        lhand_pose = smplx_param['lhand_pose'].view(len(smpl_x.joint_part['lhand']),3)
        rhand_pose = smplx_param['rhand_pose'].view(len(smpl_x.joint_part['rhand']),3)
        trans = smplx_param['trans'].view(1,3)

        # forward kinematics# 使用前向运动学计算从零姿态到图像姿态的转换矩阵
        pose = torch.cat((root_pose, body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose)) 
        pose = axis_angle_to_matrix(pose)
        _, transform_mat_joint_2 = batch_rigid_transform(pose[None,:,:,:], joint_zero_pose[None,:,:], self.smplx_layer.parents)
        transform_mat_joint_2 = transform_mat_joint_2[0]
        
        # 3. combine 1. 大 pose -> zero pose and 2. zero pose -> image pose将两个转换矩阵结合起来，得到从大姿态到图像姿态的转换矩阵
        transform_mat_joint = torch.bmm(transform_mat_joint_2, transform_mat_joint_1)
        # 返回最终的转换矩阵
        return transform_mat_joint
    
    #据每个顶点的皮肤权重和关节转换矩阵来计算顶点转换矩阵 实现关节运动到顶点位移的转换
    def get_transform_mat_vertex(self, transform_mat_joint, nn_vertex_idxs):
        # 获取每个顶点的皮肤权重
        skinning_weight = self.skinning_weight[nn_vertex_idxs,:]
        # 使用皮肤权重和关节转换矩阵计算顶点转换矩阵
        transform_mat_vertex = torch.matmul(skinning_weight, transform_mat_joint.view(smpl_x.joint_num,16)).view(smpl_x.vertex_num_upsampled,4,4)
        # 返回顶点转换矩阵
        return transform_mat_vertex
    #使用线性混合皮肤绑定（LBS）算法来根据顶点转换矩阵和平移向量来更新每个顶点的坐标
    def lbs(self, xyz, transform_mat_vertex, trans):
        # 将 xyz 坐标转换为齐次坐标（即在每个坐标后面添加 1）
        xyz = torch.cat((xyz, torch.ones_like(xyz[:,:1])),1) # 大 pose. xyz1
        # 使用顶点转换矩阵将每个顶点的坐标转换到新的位置
        xyz = torch.bmm(transform_mat_vertex, xyz[:,:,None]).view(smpl_x.vertex_num_upsampled,4)[:,:3]
        # 将转换后的坐标加上平移向量
        xyz = xyz + trans
        # 返回最终的坐标
        return xyz
    
    #从三维模型的triplane表示中提取特征
    def extract_tri_feature(self):
        # 1. 提取所有顶点的triplane特征
        # 将坐标归一化到[-1,1]
        xyz = self.pos_enc_mesh
        xyz = xyz - torch.mean(xyz,0)[None,:]# 中心化坐标
        x = xyz[:,0] / (cfg.triplane_shape_3d[0]/2)# 归一化x坐标
        y = xyz[:,1] / (cfg.triplane_shape_3d[1]/2)# 归一化y坐标
        z = xyz[:,2] / (cfg.triplane_shape_3d[2]/2)# 归一化z坐标
        
        # 从triplane中提取特征
        xy, xz, yz = torch.stack((x,y),1), torch.stack((x,z),1), torch.stack((y,z),1)
        feat_xy = F.grid_sample(self.triplane[0,None,:,:,:], xy[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled# 提取xy平面特征
        feat_xz = F.grid_sample(self.triplane[1,None,:,:,:], xz[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled# 提取xz平面特征
        feat_yz = F.grid_sample(self.triplane[2,None,:,:,:], yz[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled# 提取yz平面特征
        tri_feat = torch.cat((feat_xy, feat_xz, feat_yz)).permute(1,0) # smpl_x.vertex_num_upsampled, cfg.triplane_shape[0]*3# 合并三个平面的特征，并调整维度顺序

        # 2. 提取面部顶点的triplane特征
        # 将坐标归一化到[-1,1]
        xyz = self.pos_enc_mesh[self.is_face,:]
        xyz = xyz - torch.mean(xyz,0)[None,:]# 中心化坐标
        x = xyz[:,0] / (cfg.triplane_face_shape_3d[0]/2) # 归一化x坐标
        y = xyz[:,1] / (cfg.triplane_face_shape_3d[1]/2)# 归一化y坐标
        z = xyz[:,2] / (cfg.triplane_face_shape_3d[2]/2)# 归一化z坐标
        
         # 从triplane中提取特征
        xy, xz, yz = torch.stack((x,y),1), torch.stack((x,z),1), torch.stack((y,z),1)
        feat_xy = F.grid_sample(self.triplane_face[0,None,:,:,:], xy[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled# 提取xy平面特征
        feat_xz = F.grid_sample(self.triplane_face[1,None,:,:,:], xz[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled# 提取xz平面特征
        feat_yz = F.grid_sample(self.triplane_face[2,None,:,:,:], yz[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled# 提取yz平面特征
        tri_feat_face = torch.cat((feat_xy, feat_xz, feat_yz)).permute(1,0) # sum(self.is_face), cfg.triplane_shape[0]*3# 合并三个平面的特征，并调整维度顺序
        
        # combine 1 and 2# 结合1和2的结果
        tri_feat[self.is_face] = tri_feat_face# 将面部顶点的特征更新到总特征中
        return tri_feat# 返回最终的triplane特征
    
    

    def extract_hex_feature(self, t):
        def normalize_coords(coords, shape):
            # coords: [N, D], shape: [D]
            shape_tensor = torch.tensor(shape, device=coords.device, dtype=coords.dtype)
            return coords / (shape_tensor / 2)  # 广播执行 [N, D] / [D]

        def query_hexplane(planes, coords_2d):
            # coords_2d: [N, 2] -> [1, N, 1, 2]
            return F.grid_sample(
                planes, coords_2d[None, :, None, :], align_corners=True
            )[0, :, :, 0]  # -> [C, N]

        def get_plane_features(hexplane, coords, shape):
            coords_norm = normalize_coords(coords[:, :3], shape[:3])
            t_norm = coords[:, 3:4] / (shape[3] / 2)

            x, y, z = coords_norm[:, 0], coords_norm[:, 1], coords_norm[:, 2]
            t_ = t_norm[:, 0]

            xy = torch.stack((x, y), dim=1)
            xz = torch.stack((x, z), dim=1)
            yz = torch.stack((y, z), dim=1)
            xt = torch.stack((x, t_), dim=1)
            yt = torch.stack((y, t_), dim=1)
            zt = torch.stack((z, t_), dim=1)

            feat_xy = query_hexplane(hexplane[0:1], xy)
            feat_xz = query_hexplane(hexplane[1:2], xz)
            feat_yz = query_hexplane(hexplane[2:3], yz)
            feat_xt = query_hexplane(hexplane[3:4], xt)
            feat_yt = query_hexplane(hexplane[4:5], yt)
            feat_zt = query_hexplane(hexplane[5:6], zt)

            return torch.cat([feat_xy, feat_xz, feat_yz, feat_xt, feat_yt, feat_zt], dim=0).permute(1, 0)  # [N, C*6]

        # Step 1: 全身特征
        xyz = self.pos_enc_mesh - self.pos_enc_mesh.mean(0, keepdim=True)  # [N, 3]
        N = xyz.shape[0]

        if t.dim() == 0:
            t = t.expand(N).view(-1, 1)
        elif t.dim() == 1:
            t = t.view(-1, 1).expand(N, 1) if t.shape[0] == 1 else t.view(-1, 1)

        coords = torch.cat([xyz, t], dim=1)  # [N, 4]
        hex_feat = get_plane_features(self.hexplane, coords, cfg.hexplane_shape_3d)

        # Step 2: 面部特征
        face_xyz = self.pos_enc_mesh[self.is_face] - self.pos_enc_mesh[self.is_face].mean(0, keepdim=True)
        face_t = t[self.is_face]
        face_coords = torch.cat([face_xyz, face_t], dim=1)
        hex_feat_face = get_plane_features(self.hexplane_face, face_coords, cfg.hexplane_face_shape_4d)

        hex_feat[self.is_face] = hex_feat_face
        return hex_feat

    # #接受三平面特征 tri_feat 和 SMPL-X 参数 smplx_param 作为输入，并输出高斯分布的均值偏移和尺度偏移。这些输出可以用于后续的几何处理或渲染任务
    # def forward_geo_network(self, tri_feat, smplx_param):
    #     # poses from smplx parameters # 从SMPL-X参数中提取身体姿势
    #     body_pose = smplx_param['body_pose'].view(len(smpl_x.joint_part['body'])-1,3)
    #     # 提取颚部姿势
    #     jaw_pose = smplx_param['jaw_pose'].view(1,3)
    #     # 提取左眼姿势
    #     leye_pose = smplx_param['leye_pose'].view(1,3)
    #     # 提取右眼姿势
    #     reye_pose = smplx_param['reye_pose'].view(1,3)
    #     # 提取左手姿势
    #     lhand_pose = smplx_param['lhand_pose'].view(len(smpl_x.joint_part['lhand']),3)
    #     # 提取右手姿势
    #     rhand_pose = smplx_param['rhand_pose'].view(len(smpl_x.joint_part['rhand']),3)

    #     # combine pose with triplane feature# 将所有姿势组合成一个张量
    #     pose = torch.cat((body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose))
    #     # 将轴角姿势转换为旋转矩阵，然后转换为旋转6D表示，并重复以匹配上采样顶点的数量
    #     pose = matrix_to_rotation_6d(axis_angle_to_matrix(pose)).view(1,smpl_x.joint_num-1,6).repeat(smpl_x.vertex_num_upsampled,1,1) # without root pose
    #     # 将姿势张量重塑为二维张量
    #     pose = pose.view(smpl_x.vertex_num_upsampled, (smpl_x.joint_num-1)*6)
    #     # 将三平面特征与姿势特征连接起来
    #     feat = torch.cat((tri_feat, pose.detach()),1)

    #     # forward to geometry networks# 将特征前向传播到几何网络以获取几何偏移特征   这三个全不需要了，注释掉了
    #     geo_offset_feat = self.geo_offset_net(feat)
    #     # 计算高斯分布的姿势依赖均值偏移
    #     mean_offset_offset = self.mean_offset_offset_net(geo_offset_feat) # pose-dependent mean offset of Gaussians
    #     # 计算高斯分布的姿势依赖尺度偏移
    #     scale_offset = self.scale_offset_net(geo_offset_feat) # pose-dependent scale of Gaussians
    #     # 返回计算得到的均值偏移和尺度偏移
    #     return mean_offset_offset, scale_offset
    
    # #接受 SMPL-X 参数 smplx_param 和预先计算的 mean_offset_offset 作为输入，并输出调整后的偏移量
    
    
    # def get_mean_offset_offset(self, smplx_param, mean_offset_offset):
    #     # poses from smplx parameters # 从SMPL-X参数中提取身体姿势
    #     body_pose = smplx_param['body_pose'].view(len(smpl_x.joint_part['body'])-1,3)
    #     jaw_pose = smplx_param['jaw_pose'].view(1,3)
    #     leye_pose = smplx_param['leye_pose'].view(1,3)
    #     reye_pose = smplx_param['reye_pose'].view(1,3)
    #     lhand_pose = smplx_param['lhand_pose'].view(len(smpl_x.joint_part['lhand']),3)
    #     rhand_pose = smplx_param['rhand_pose'].view(len(smpl_x.joint_part['rhand']),3)
    #     pose = torch.cat((body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose)) # without root pose # 将所有姿势组合成一个张量，不包括根姿势

    #     # smplx pose-dependent vertex offset# 将轴角姿势转换为旋转矩阵，并减去单位矩阵，然后将其重塑为一维张量
    #     pose = (axis_angle_to_matrix(pose) - torch.eye(3)[None,:,:].float().cuda()).view(1,(smpl_x.joint_num-1)*9)
    #     # 计算SMPL-X姿势依赖的顶点偏移
    #     smplx_pose_offset = torch.matmul(pose.detach(), self.pose_dirs).view(smpl_x.vertex_num_upsampled,3)

    #     # 将其与回归的mean_offset_offset结合
    #     # 对于脸部和手部，使用SMPL-X偏移
    #     mask = ((self.is_rhand + self.is_lhand + self.is_face_expr) > 0)[:,None].float()
    #     mean_offset_offset = mean_offset_offset * (1 - mask)
    #     smplx_pose_offset = smplx_pose_offset * mask
    #     # 计算最终输出na'wo
    #     output = mean_offset_offset + smplx_pose_offset
    #     # 返回最终输出和mean_offset_offset
    #     return output, mean_offset_offset



    def get_mean_offset_offset(self, smplx_param, mean_offset_offset):
        # 提取 SMPL-X 参数
        body_pose = smplx_param['body_pose'].view(len(smpl_x.joint_part['body'])-1, 3)
        jaw_pose = smplx_param['jaw_pose'].view(1, 3)
        leye_pose = smplx_param['leye_pose'].view(1, 3)
        reye_pose = smplx_param['reye_pose'].view(1, 3)
        lhand_pose = smplx_param['lhand_pose'].view(len(smpl_x.joint_part['lhand']), 3)
        rhand_pose = smplx_param['rhand_pose'].view(len(smpl_x.joint_part['rhand']), 3)
        
        # 将所有姿势组合成一个张量（不包括根姿势）
        pose = torch.cat((body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose))

        # SMPL-X 姿势依赖的顶点偏移（Pose-dependent vertex offset）
        pose = (axis_angle_to_matrix(pose) - torch.eye(3)[None, :, :].float().cuda()).view(1, (smpl_x.joint_num - 1) * 9)
        smplx_pose_offset = torch.matmul(pose.detach(), self.pose_dirs).view(smpl_x.vertex_num_upsampled, 3)

        # 将其与回归的 mean_offset_offset 结合
        mask = ((self.is_rhand + self.is_lhand + self.is_face_expr) > 0)[:, None].float()
        mean_offset_offset = mean_offset_offset * (1 - mask)
        smplx_pose_offset = smplx_pose_offset * mask

        # **加入 MANO 的手部残差**
        
        # 加载 MANO 和 SMPL-X 顶点映射
        with open(osp.join('/home/suyuze/workspace/ExAvatar_mine/common/utils/human_model_files', 'smplx', 'MANO_SMPLX_vertex_ids.pkl'), 'rb') as f:
            hand_vertex_idx = pickle.load(f, encoding='latin1')

        # hand_vertex_idx['right_hand']['mano'] 
        # hand_vertex_idx['right_hand']['smplx'] 
        # hand_vertex_idx['left_hand']['mano'] 
        # hand_vertex_idx['left_hand']['smplx'] 

        # 生成 MANO 的手部 mesh，并计算其残差（delta）
        def apply_mano_offsets_to_smplx(smplx_pose_offset, mano_left_delta, mano_right_delta, hand_vertex_idx):
            for side, mano_delta in zip(['left_hand', 'right_hand'], [mano_left_delta, mano_right_delta]):
                # hand_vertex_idx[side] 是一个索引列表，直接转为 LongTensor
                smplx_ids = torch.tensor(hand_vertex_idx[side], dtype=torch.long, device=smplx_pose_offset.device)

                # 确保 mano_delta 在相同设备上，并转换为 torch.Tensor
                mano_delta = torch.tensor(mano_delta, dtype=torch.float32, device=smplx_pose_offset.device)

                # 检查索引是否越界
                assert smplx_ids.max().item() < smplx_pose_offset.shape[0], f"Error: smplx_ids 超过了 smplx_pose_offset 的最大索引值！"
                assert smplx_ids.shape[0] == mano_delta.shape[0], f"Error: smplx_ids 和 mano_delta 的形状不匹配！"

                # 替换 SMPL-X 手部区域的偏移量
                smplx_pose_offset[smplx_ids] = mano_delta

            return smplx_pose_offset

        
        # def save_mano_mesh_as_ply(verts, faces, save_path):
        #     """
        #     verts: torch.Tensor or np.ndarray of shape (N, 3)
        #     faces: np.ndarray of shape (F, 3), from MANO model
        #     save_path: str, full path to save the .ply file
        #     """
        #     if isinstance(verts, torch.Tensor):
        #         verts = verts.detach().cpu().numpy()

        #     mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        #     mesh.export(save_path)

        mano_right_verts = self.mano_right_layer(
            hand_pose=rhand_pose.reshape(1, -1),
            global_orient=smplx_param['root_pose'].reshape(1, -1),
            betas=self.shape_param[:10].view(1, -1) 
        ).vertices[0]  # 获取右手 MANO 的 mesh
        # 假设你已经加载过 MANO 模型，且含有 face 信息：
        # mano_faces = self.mano_right_layer.faces  # shape: (Nf, 3)

        # 保存为 ply
        # save_mano_mesh_as_ply(mano_right_verts, mano_faces, "mano_right_test.ply")


        mano_left_verts = self.mano_left_layer(
            hand_pose=lhand_pose.reshape(1, -1),
            global_orient=smplx_param['root_pose'].reshape(1, -1),
            betas=self.shape_param[:10].view(1, -1) 
        ).vertices[0]  # 获取左手 MANO 的 mesh

        mano_right_delta = mano_right_verts - self.mano_right_layer.v_template.to(mano_right_verts.device)
        mano_left_delta = mano_left_verts - self.mano_left_layer.v_template.to(mano_left_verts.device)

        # 将 MANO 残差转移到 SMPL-X 上
        smplx_pose_offset = apply_mano_offsets_to_smplx(
            smplx_pose_offset,
            mano_left_delta,
            mano_right_delta,
            hand_vertex_idx
        )
        
        is_hand = ((self.is_rhand + self.is_lhand) > 0).float()  # [V]
        # hand_mask = is_hand[:, None]  # [V, 1]
        hand_pose_offset = smplx_pose_offset[is_hand.bool()]  # [N_hand, 3]
        hand_cond = torch.cat([
            smplx_param['lhand_pose'].view(-1),  # [15]
            smplx_param['rhand_pose'].view(-1),  # [15]
        ], dim=0)  # [30]
        hand_cond = hand_cond.unsqueeze(0).repeat(hand_pose_offset.shape[0], 1)  # [N_hand, 30]
        mlp_input = torch.cat([hand_pose_offset, hand_cond], dim=1)  # [N_hand, 33]
        
        
        hand_refined = self.mlp_mano(mlp_input)
        smplx_pose_offset[is_hand.bool()] = hand_refined
        # 计算最终的输出
        output = mean_offset_offset + smplx_pose_offset

        # 返回最终输出
        return output, mean_offset_offset

    # 它首先从 SMPL-X 参数中提取各种姿势，然后计算这些姿势对应的旋转6D表示。接着，它计算每个顶点在世界坐标系中的法线，并处理腔体部分的法线方向。
    # 最后，它将三平面特征、姿势特征和法线特征连接起来，并将这些特征前向传播到 RGB 网络以获取 RGB 偏移
    #它接受三平面特征 tri_feat、SMPL-X 参数 smplx_param 和顶点坐标 xyz 作为输入，并输出 RGB 偏移

    def forward_rgb_network(self, tri_feat, smplx_param, xyz):
        # poses from smplx parameters# 从SMPL-X参数中提取身体姿势
        root_pose =smplx_param['root_pose'].view(1,3)
        body_pose = smplx_param['body_pose'].view(len(smpl_x.joint_part['body'])-1,3)
        jaw_pose = smplx_param['jaw_pose'].view(1,3)
        leye_pose = smplx_param['leye_pose'].view(1,3)
        reye_pose = smplx_param['reye_pose'].view(1,3)
        lhand_pose = smplx_param['lhand_pose'].view(len(smpl_x.joint_part['lhand']),3)
        rhand_pose = smplx_param['rhand_pose'].view(len(smpl_x.joint_part['rhand']),3)
        
        # transform root pose from camera coordinate system to world coordinate system# 将所有姿势组合成一个张量
        pose = torch.cat((root_pose,body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose))
        pose = matrix_to_rotation_6d(axis_angle_to_matrix(pose))# 将轴角姿势转换为旋转矩阵，然后转换为旋转6D表示
        # 将姿势张量重塑并重复以匹配上采样顶点的数量
        # pose = pose.view(1,(smpl_x.joint_num-1)*6).repeat(smpl_x.vertex_num_upsampled,1) # smpl_x.vertex_num_upsampled, (smpl_x.joint_num-1)*6
        # self.skinning_weight[:, 1:] 这个改动加了rootpose，要是没用再删掉
        pose = torch.matmul(self.skinning_weight, pose) 
        # per-vertex normal in world coordinate system# 计算每个顶点在世界坐标系中的法线
        with torch.no_grad():
            normal = Meshes(verts=xyz[None], faces=torch.LongTensor(smpl_x.face_upsampled).cuda()[None]).verts_normals_packed().reshape(smpl_x.vertex_num_upsampled,3)
            is_cavity = self.is_cavity[:,None].float()
            normal = normal * (1 - is_cavity) + (-normal) * is_cavity # cavity has opposite normal direction in the template mesh# 腔体部分的法线方向相反

        # forward to rgb network# 将三平面特征、姿势特征和法线特征连接起来
        feat = torch.cat((tri_feat, pose.detach(), normal.detach()),1)
        # 将特征前向传播到RGB网络以获取RGB偏移,MLP
        
        rgb_offset = self.rgb_offset_net(feat) # pose-dependent rgb offset of Gaussians# 姿势依赖的RGB偏移
        # 返回RGB偏移
        return rgb_offset

    def lr_idx_to_hr_idx(self, idx):
        # follow 'subdivide_homogeneous' function of https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/subdivide_meshes.html#SubdivideMeshes
        # the low-res part takes first N_lr vertices out of N_hr vertices
        return idx



    # time 单位是秒
    def forward(self, smplx_param, cam_param=None, is_world_coord=True,t = None):

        # print('----------------------------------------------------------------------------------')
        # print(self.is_face_expr.sum().item())
        # print(self.is_lhand.sum().item())
        # print(self.is_rhand.sum().item())
        # print('----------------------------------------------------------------------------------')
        # 获取人体的中性姿态，返回对应的网格、未上采样的中性姿态、其他信息以及变换矩阵
        mesh_neutral_pose, mesh_neutral_pose_wo_upsample,joint_neutral_pose, transform_mat_neutral_pose = self.get_neutral_pose_human(jaw_zero_pose=True, use_id_info=True)
        # 获取关节的零姿态
        joint_zero_pose = self.get_zero_pose_human()


        t = (t - 2.5) / 2.5          # 中心化 + 归一化到 [-1, 1]，2.5是时间中心
        hex_feat = self.extract_hex_feature(t)
        hex_feat1 = self.hex_to_tri(hex_feat) # 维度对齐
        
        # extract triplane feature# 提取三平面特征
        tri_feat = self.extract_tri_feature()

        concat_feat = torch.cat([tri_feat, hex_feat1], dim=1)  # [N, 2*C*3]
        
        alpha = self.alpha_mlp(concat_feat)
        fused_feat = tri_feat * (1 - alpha) + hex_feat1 * alpha  # [N, C*3]
        
      
        # get Gaussian assets获取高斯特征
        geo_feat = self.geo_net(fused_feat) # 利用提取的三平面特征得到几何特征 将 tri_feat 转换为几何特征（geo_feat），这里的 geo_feat 通常包含了与物体的几何形状相关的更多高层次信息，比如体积、形状、曲率等
        mean_offset = self.mean_offset_net(geo_feat) # mean offset of Gaussians# 高斯平面的均值偏移
        scale = self.scale_net(geo_feat) # scale of Gaussians # 高斯平面的缩放参数
        rgb = self.rgb_net(fused_feat) # rgb of Gaussians# 高斯平面的RGB颜色
        mean_3d = mesh_neutral_pose + mean_offset # 大 pose# 计算初步的大姿态
        # mean_offset_offset, scale_offset = self.forward_geo_network(tri_feat, smplx_param)
        # smplx_param 中的各部分 joint pose（axis-angle 格式）
        pose = torch.cat([
            smplx_param['root_pose'].view(1,3),
            smplx_param['body_pose'].view(len(smpl_x.joint_part['body'])-1,3),  # [21, 3]
            smplx_param['jaw_pose'].view(1,3),   # [1, 3]
            smplx_param['leye_pose'].view(1,3),  # [1, 3]
            smplx_param['reye_pose'].view(1,3),  # [1, 3]
            smplx_param['lhand_pose'].view(len(smpl_x.joint_part['lhand']),3), # [15, 3]
            smplx_param['rhand_pose'].view(len(smpl_x.joint_part['rhand']),3)  # [15, 3]
        ], dim=0)  # [55, 3]

        # 转换为 rot6d 表示
        joint_rot6d = matrix_to_rotation_6d(axis_angle_to_matrix(pose))  # [55, 6]
        ###___________________normal________________________
        with torch.no_grad():
            normal_0 = Meshes(verts=mean_3d[None], faces=torch.LongTensor(smpl_x.face_upsampled).cuda()[None]).verts_normals_packed().reshape(smpl_x.vertex_num_upsampled,3)
            is_cavity = self.is_cavity[:,None].float()
            normal_0 = normal_0 * (1 - is_cavity) + (-normal_0) * is_cavity # cavity has opposite normal direction in the template mesh# 腔体部分的法线方向相反



        '''
        # get pose-dependent Gaussian assets # 获取与姿态相关的高斯特征
        mean_offset_offset, scale_offset = self.forward_geo_network(fused_feat, 
                                                                    self.skinning_weight.argmax(dim=1).long(),
                                                                    joint_rot6d,
                                                                    joint_neutral_pose#55
                                                                    )# 与姿态相关的偏移和缩放----------------------------------MLP1
        '''

        mean_offset_offset, scale_offset = self.forward_geo_network(
                                                                    fused_feat, 
                                                                    self.skinning_weight.argmax(dim=1).long(),
                                                                    joint_rot6d, 
                                                                    joint_neutral_pose, 
                                                                    normal_0  # <--- 这里对接
                                                                )
        scale, scale_refined = torch.exp(scale).repeat(1,3), torch.exp(scale+scale_offset).repeat(1,3)# 将缩放参数转化为正值

        
        mean_combined_offset, mean_offset_offset = self.get_mean_offset_offset(smplx_param, mean_offset_offset)# 合并偏移
        
            
        mean_3d_refined = mean_3d + mean_combined_offset # 大 pose # 计算经过调整的大姿态

        # smplx facial expression offset# 处理SMPL-X的面部表情偏移
        # print('*********************************************')
        # print(smplx_param['expr'][None,None,:][:,:,:10].shape,self.expr_dirs.shape)

        smplx_expr_offset = (smplx_param['expr'][None,None,:] * self.expr_dirs).sum(2) # 计算面部表情的偏移
        mean_3d = mean_3d + smplx_expr_offset # 大 pose# 更新大姿态以包含面部表情偏移
        
        mean_3d_refined = mean_3d_refined + smplx_expr_offset # 大 pose# 更新经过调整的大姿态

        # get nearest vertex # 获取最近的顶点
        # for hands and face, assign original vertex index to use sknning weight of the original vertex
         # 在网格中找到与大姿态最近的顶点 
         # nn_vertex_idxs 将会是一个数组，包含 mean_3d 中每个点在 mesh_neutral_pose_wo_upsample 中的最近邻的一个点的索引
        nn_vertex_idxs = knn_points(mean_3d[None,:,:], mesh_neutral_pose_wo_upsample[None,:,:], K=1, return_nn=True).idx[0,:,0] # dimension: smpl_x.vertex_num_upsampled
         # 将低分辨率索引转换为高分辨率索引
        nn_vertex_idxs = self.lr_idx_to_hr_idx(nn_vertex_idxs)
        # 创建一个掩码以识别手和脸的顶点
        mask = (self.is_rhand + self.is_lhand + self.is_face) > 0
       # 方便手和脸的顶点保留原来的顶点索引
        nn_vertex_idxs[mask] = torch.arange(smpl_x.vertex_num_upsampled).cuda()[mask]

        # get transformation matrix of the nearest vertex and perform lbs# 获取最近顶点的变换矩阵并执行线性加权混合（LBS）
        transform_mat_joint = self.get_transform_mat_joint(transform_mat_neutral_pose, joint_zero_pose, smplx_param)# 计算关节的变换矩阵
        transform_mat_vertex = self.get_transform_mat_vertex(transform_mat_joint, nn_vertex_idxs)# 基于关节变换和最近顶点索引计算顶点变换矩阵
        mean_3d = self.lbs(mean_3d, transform_mat_vertex, smplx_param['trans']) # posed with smplx_param# 进行线性加权混合以获取姿态
        mean_3d_refined = self.lbs(mean_3d_refined, transform_mat_vertex, smplx_param['trans']) # posed with smplx_param# 进行线性加权混合以获取经过调整的姿态


        #  # 从相机坐标系转换到世界坐标系
        if not is_world_coord:
             # 转换到世界坐标系
            mean_3d = torch.matmul(torch.inverse(cam_param['R']), (mean_3d - cam_param['t'].view(1,3)).permute(1,0)).permute(1,0)
            mean_3d_refined = torch.matmul(torch.inverse(cam_param['R']), (mean_3d_refined - cam_param['t'].view(1,3)).permute(1,0)).permute(1,0)
        # forward to rgb network# 将经过调整的姿态输入到RGB网络中
        rgb_offset = self.forward_rgb_network(fused_feat, smplx_param, mean_3d_refined)# 计算RGB偏移   -----------------------------------------------MLP2

        rgb, rgb_refined = (torch.tanh(rgb) + 1) / 2, (torch.tanh(rgb + rgb_offset) + 1) / 2 # normalize to [0,1]# 将RGB输出归一化到[0,1]
        
        ## 计算高斯和偏移量
        rotation = matrix_to_quaternion(torch.eye(3).float().cuda()[None,:,:].repeat(smpl_x.vertex_num_upsampled,1,1)) # # 常量旋转（单位矩阵转换为四元数）
        opacity = torch.ones((smpl_x.vertex_num_upsampled,1)).float().cuda() # constant opacity # 常量透明度
        

        assets = { # 将所有资产组合到字典中
                'mean_3d': mean_3d, 
                'opacity': opacity, 
                'scale': scale, 
                'rotation': rotation, 
                'rgb': rgb_refined
                # 'face':residual
                
                
        }
        assets_refined = { # 经调整的资产组合到字典中
                'mean_3d': mean_3d_refined, 
                'opacity': opacity, 
                'scale': scale_refined, 
                'rotation': rotation, 
                'rgb': rgb_refined
                # 'face':residual
                
                }
        offsets = { # 拿到偏移量信息
                'mean_offset': mean_offset,
                'mean_offset_offset': mean_offset_offset,
                'scale_offset': scale_offset,
                'rgb_offset': rgb_offset
                
                }
        # 返回最终的资产、调整后的资产、偏移量及中性网格
        return assets, assets_refined, offsets, mesh_neutral_pose
# 定义一个 GaussianRenderer 类，继承自 nn.Module，用于渲染高斯分布图像。
class GaussianRenderer(nn.Module):
    
    
    def __init__(self):
        # 初始化方法。
        super(GaussianRenderer, self).__init__()
        # 调用父类的初始化方法，以确保模块正确初始化。

    # 核心部分是 forward 方法，它处理输入的高斯分布数据，并使用光栅化方法将其渲染成2D图像
    def forward(self, gaussian_assets, img_shape, cam_param, bg=None):
        # 定义前向传播方法，它接受高斯渲染所需的参数。
        # gaussian_assets: 包含高斯分布的资产（如均值、透明度、缩放等）的字典。
        # img_shape: 图像的形状，用于渲染输出的大小。
        # cam_param: 相机参数字典，包含旋转、平移等信息。
        # bg: 可选的背景色，如果未提供，则默认值为 None。
        #gaussian_assets 是一个包含所需高斯分布信息的字典
        mean_3d = gaussian_assets['mean_3d']    #  从高斯资产中提取 3D 均值。
        opacity = gaussian_assets['opacity']      # 提取透明度信息。
        scale = gaussian_assets['scale']          # 提取缩放信息。
        rotation = gaussian_assets['rotation']    # 提取旋转信息。
        rgb = gaussian_assets['rgb']              # 提取 RGB 颜色信息。
                   

        # 创建光栅化器 参数准备与相机设置
        # 使用以下配置调整 view_matrix 和 proj_matrix 的排列方式，参考链接：
        # https://github.com/graphdeco-inria/gaussian-splatting/blob/.../cameras.py#L54
        # https://github.com/graphdeco-inria/gaussian-splatting/blob/.../cameras.py#L55
        
        fov = get_fov(cam_param['focal'], cam_param['princpt'], img_shape)  # 计算视场角。
        view_matrix = get_view_matrix(cam_param['R'], cam_param['t']).permute(1,0)  # 获取视图矩阵并调整维度顺序。
        proj_matrix = get_proj_matrix(cam_param['focal'], cam_param['princpt'], img_shape, 0.01, 100, 1.0).permute(1,0)  # 获取投影矩阵并调整维度顺序。
        full_proj_matrix = torch.mm(view_matrix, proj_matrix)  # 计算完整的投影矩阵。
        cam_pos = view_matrix.inverse()[3,:3]  # 从视图矩阵计算相机位置。
        
        if bg is None:
            bg = torch.ones((3)).float().cuda()  # 如果没有背景色，则默认背景为白色。

        # 光栅化设置，包括图像的高度和宽度、视图矩阵等参数。
        raster_settings = GaussianRasterizationSettings(
            image_height=img_shape[0],                # 图像高度
            image_width=img_shape[1],                  # 图像宽度
            tanfovx=float(torch.tan(fov[0]/2)),       # 水平视场的切值
            tanfovy=float(torch.tan(fov[1]/2)),       # 垂直视场的切值
            bg=bg,                                      # 背景色
            scale_modifier=1.0,                         # 尺寸调整因子
            viewmatrix=view_matrix,                     # 视图矩阵
            projmatrix=full_proj_matrix,                # 完整的投影矩阵
            sh_degree=0,                                # 模拟化度（dummy），因为 RGB 值已计算
            campos=cam_pos,                             # 相机位置
            prefiltered=False,                           # 不使用预过滤
            debug=False                                  # 非调试模式
        )
        #这是进行高斯光栅化的关键类
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)  # 初始化 GaussianRasterizer，使用上述设置。
        
        # 准备一个2D均值张量，以便在渲染过程中进行梯度计算和反向传播
        point_num = mean_3d.shape[0]  # 获取 3D 均值的数量。
        mean_2d = torch.zeros((point_num,3)).float().cuda()  # 创建一个 2D 均值的张量，并初始化为零。
        mean_2d.requires_grad = True  # 允许 mean_2d 张量自动计算梯度。
        mean_2d.retain_grad()  # 保留 mean_2d 的梯度，以便后续使用。
        
        # 将可见的高斯体渲染到图像中，并获取它们在屏幕上的半径。
        render_img, radius, render_depthmap, render_mask= rasterizer(
            means3D=mean_3d,          # 3D 均值
            means2D=mean_2d,          # 2D 均值
            shs=None,                 # 球谐基，留空
            colors_precomp=rgb,       # 预计算的颜色
            opacities=opacity,        # 透明度
            scales=scale,             # 缩放
            rotations=rotation,       # 旋转
            cov3D_precomp=None)      # 预计算的协方差，留空
        
        # 返回渲染结果，包括图像、深度图、掩码等数据。
        return {
            'img': render_img,         # 渲染的图像
            'depthmap': render_depthmap, # 渲染的深度图
            'mask': render_mask,       # 渲染的掩码
            'mean_2d': mean_2d,       # 计算的 2D 均值
            'is_vis': radius > 0,     # 可视性标记，判断半径是否大于 0
            'radius': radius           # 计算的半径
        }

# 定义一个 SMPLXParamDict 类，继承自 nn.Module，用于管理和优化 SMPL-X 模型的参数。
class SMPLXParamDict(nn.Module):
    
    
    def __init__(self):
        # 初始化方法。
        super(SMPLXParamDict, self).__init__()  # 调用父类的初始化方法，确保模块正确初始化。

    # 初始化所有帧的 SMPL-X 参数
    # 用于从头开始训练模型
    def init(self, smplx_params):
        # `init` 方法接收 SMPL-X 参数并初始化模型的参数字典。
        _smplx_params = {}  # 创建一个空字典以存储参数。
        
        for capture_id in smplx_params.keys():
            # 遍历每个捕获的 ID。
            _smplx_params[capture_id] = nn.ParameterDict({})  # 为每个捕获创建一个参数字典。
            
            for frame_idx in smplx_params[capture_id].keys():
                # 遍历每个捕获中每一帧的索引。
                _smplx_params[capture_id][str(frame_idx)] = nn.ParameterDict({})  # 为每一帧创建一个参数字典。
                
                for param_name in ['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose', 'expr', 'trans']:
                    # 遍历每个参数名称。
                    if 'pose' in param_name:
                        # 如果参数名称中包含 'pose'，则将其转换为 6D 旋转表示。
                        _smplx_params[capture_id][str(frame_idx)][param_name] = nn.Parameter(
                            matrix_to_rotation_6d(axis_angle_to_matrix(smplx_params[capture_id][frame_idx][param_name].cuda()))
                        )
                    else:
                        # 对于非姿态参数，直接将其转为 PyTorch 的可训练参数。
                        _smplx_params[capture_id][str(frame_idx)][param_name] = nn.Parameter(smplx_params[capture_id][frame_idx][param_name].cuda())
        
        self.smplx_params = nn.ParameterDict(_smplx_params)  # 将初始化的参数字典存储为类的成员变量。

    def get_optimizable_params(self):
        # 获取可优化的参数列表。
        optimizable_params = []  # 创建一个空列表以存储可优化的参数。
        
        for capture_id in self.smplx_params.keys():
            # 遍历每个捕获的 ID。
            for frame_idx in self.smplx_params[capture_id].keys():
                # 遍历每个帧的索引。
                for param_name in self.smplx_params[capture_id][frame_idx].keys():
                    # 将每个参数添加到可优化参数列表中，包含参数、名称和学习率。
                    optimizable_params.append({
                        'params': [self.smplx_params[capture_id][frame_idx][param_name]], 
                        'name': 'smplx_' + param_name + '_' + capture_id + '_' + frame_idx, 
                        'lr': cfg.smplx_param_lr
                    })
        
        return optimizable_params  # 返回可优化的参数列表。

    def forward(self, capture_ids, frame_idxs):
        # 定义前向传播方法，接受多个捕获 ID 和帧索引。
        out = []  # 创建一个空列表以存储输出参数字典。
        
        for capture_id, frame_idx in zip(capture_ids, frame_idxs):
            # 遍历每个捕获 ID 和对应的帧索引。
            capture_id = str(capture_id)  # 将捕获 ID 转为字符串以进行索引。
            frame_idx = str(int(frame_idx))  # 将帧索引转为字符串以进行索引。
            smplx_param = {}  # 创建一个空字典以存储 SMPL-X 参数。
            
            for param_name in self.smplx_params[capture_id][frame_idx].keys():
                # 遍历当前帧的所有 SMPL-X 参数。
                if 'pose' in param_name:
                    # 如果参数名称中包含 'pose'，则转换为轴角表示。
                    smplx_param[param_name] = matrix_to_axis_angle(rotation_6d_to_matrix(self.smplx_params[capture_id][frame_idx][param_name]))
                else:
                    # 对于非姿态参数，直接存储相应的值。
                    smplx_param[param_name] = self.smplx_params[capture_id][frame_idx][param_name]
            out.append(smplx_param)  # 将当前帧的 SMPL-X 参数字典添加到输出列表。
        
        return out  # 返回所有帧的 SMPL-X 参数列表。

