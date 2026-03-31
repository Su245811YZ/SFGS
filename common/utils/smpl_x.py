import sys
import numpy as np
import torch
from torch.nn import functional as F
import os.path as osp
from config import cfg
from utils.smplx import smplx
import pickle
from pytorch3d.structures import Meshes
from pytorch3d.ops import SubdivideMeshes
from smplx.lbs import batch_rigid_transform
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
import math
from scipy.spatial import cKDTree
#用基础的通用参数初始化模型
class SMPLX(object):
    def __init__(self):
        #  # 定义形状和表情参数的维度
        self.device = torch.device('cuda')
        self.shape_param_dim = 100 
        # 50改成10了
        self.expr_param_dim = 50
        # 定义SMPL-X层的参数
        self.layer_arg = {'create_global_orient': False, 'create_body_pose': False, 
                          'create_left_hand_pose': False, 'create_right_hand_pose': False, 
                          'create_jaw_pose': False, 'create_leye_pose': False, 'create_reye_pose': False, 
                          'create_betas': False, 'create_expression': False, 'create_transl': False}
        # 创建SMPL-X层
        self.layer = {  # 初始化一个字典，存储不同性别的模型实例。
                gender: smplx.create(  # 设置字典的键为性别，值为调用 smplx.create 创建的模型实例。
                    cfg.human_model_path,  # 指定模型文件的路径，从配置中获取。
                    'smplx',  # 模型类型为 'smplx'。
                    gender=gender,  # 当前模型的性别，由字典的键提供。
                    num_betas=self.shape_param_dim,  # 设置形状参数的维度，使用实例中的 shape_param_dim。
                    num_expression_coeffs=self.expr_param_dim,  # 设置表情系数的维度，使用实例中的 expr_param_dim。
                    use_pca=False,  # 禁用主成分分析（PCA）。
                    use_face_contour=True,  # 启用面部轮廓信息。
                    **self.layer_arg  # 解包其他参数，这些参数可能进一步自定义模型的创建过程。
                        )
                for gender in ['neutral', 'male', 'female']  # 遍历这三个性别，生成相应的模型实例。
        }
        # 加载脸部顶点索引
        self.face_vertex_idx = np.load(osp.join(cfg.human_model_path, 'smplx', 'SMPL-X__FLAME_vertex_ids.npy'))
         # 更新层以包含FLAME模型的表情参数
        self.layer = {gender: self.get_expr_from_flame(self.layer[gender]) for gender in ['neutral', 'male', 'female']}
        # 定义顶点数量
        self.vertex_num = 10475
        # 获取原始脸部信息
        self.face_orig = self.layer['neutral'].faces.astype(np.int64)
        # 添加腔体并获取更新后的脸部信息
        self.is_cavity, self.face = self.add_cavity()
         # 加载手部顶点索引
        with open(osp.join(cfg.human_model_path, 'smplx', 'MANO_SMPLX_vertex_ids.pkl'), 'rb') as f:
            hand_vertex_idx = pickle.load(f, encoding='latin1')
        
        self.rhand_vertex_idx = hand_vertex_idx['right_hand']
        self.lhand_vertex_idx = hand_vertex_idx['left_hand']
        # 获取表情顶点索引
        self.expr_vertex_idx = self.get_expr_vertex_idx()

        # SMPLX joint set# 定义SMPL-X关节集
        self.joint_num = 55 # 22 (body joints) + 3 (face joints) + 30 (hand joints)
        self.joints_name = \
        ( # 身体关节
            'Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', # body joints
        # 脸部关节
        'Jaw', 'L_Eye', 'R_Eye', # face joints
        # 左手关节
        'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', # left hand joints
        # 右手关节
        'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3' # right hand joints
        )
        # 找到并存储根关节（'Pelvis'）在关节名称列表 self.joints_name 中的索引位置
        self.root_joint_idx = self.joints_name.index('Pelvis')
        #self.joint_part，用于存储不同身体部位关节的索引范围
        self.joint_part = \
        {'body': range(self.joints_name.index('Pelvis'), self.joints_name.index('R_Wrist')+1),
        'face': range(self.joints_name.index('Jaw'), self.joints_name.index('R_Eye')+1),
        'lhand': range(self.joints_name.index('L_Index_1'), self.joints_name.index('L_Thumb_3')+1),
        'rhand': range(self.joints_name.index('R_Index_1'), self.joints_name.index('R_Thumb_3')+1)}
        # 定义中立姿态下的身体和下巴姿态
        self.neutral_body_pose = torch.zeros((len(self.joint_part['body'])-1,3)) # 大 pose in axis-angle representation (body pose without root joint)
        self.neutral_body_pose[0] = torch.FloatTensor([0, 0, 1])
        self.neutral_body_pose[1] = torch.FloatTensor([0, 0, -1])
        self.neutral_jaw_pose = torch.FloatTensor([1/3, 0, 0])
        
        # subdivider # 子划分器
        self.subdivider_list = self.get_subdivider(2)
        self.face_upsampled = self.subdivider_list[-1]._subdivided_faces.cpu().numpy()
        self.vertex_num_upsampled = int(np.max(self.face_upsampled)+1)
        #        # print("上采样后顶点数:", self.vertex_num_upsampled)#上采样后顶点数: 167390
    #从FLAME模型创建一个人类模型并返回这个模型的状态
    def get_expr_from_flame(self, smplx_layer):
        # # 创建一个FLAME模型的层，使用给定的配置文件路径、人类模型参数和性别
        flame_layer = smplx.create(cfg.human_model_path, 
                                   'flame', 
                                   gender='neutral', 
                                   num_betas=100, 
                                   num_expression_coeffs=self.expr_param_dim)
         # 将FLAME模型的表情方向（expr_dirs）复制到smplx_layer的相应索引位置
        smplx_layer.expr_dirs[self.face_vertex_idx,:,:] = flame_layer.expr_dirs
        # 返回更新后的smplx_layer
        return smplx_layer

    def set_id_info(self, shape_param, face_offset, joint_offset, locator_offset):
        # 设置身份信息，包括形状参数、脸部偏移、关节偏移和定位器偏移
        self.shape_param = shape_param
        self.face_offset = face_offset
        self.joint_offset = joint_offset
        self.locator_offset = locator_offset

    def get_joint_offset(self, joint_offset):
        # 创建一个全为1的权重张量，形状为 (1, joint_num, 1)
        weight = torch.ones((1,self.joint_num,1)).float().to(self.device)
        # 将根关节的权重设置为0，因为根关节通常不进行偏移
        weight[:,self.root_joint_idx,:] = 0
        # 将关节偏移与权重相乘，以排除根关节的偏移
        joint_offset = joint_offset * weight
        # 返回处理后的关节偏移
        return joint_offset
    #2
    def get_subdivider(self, subdivide_num):
        # 获取中性性别模型的顶点模板，并转换为浮点型和CUDA张量
        vert = self.layer['neutral'].v_template.float().to(self.device)
        # 将脸部信息转换为长整型CUDA张量
        face = torch.LongTensor(self.face).to(self.device)
        # 创建一个Meshes对象，包含顶点和脸部信息
        mesh = Meshes(vert[None,:,:], face[None,:,:])
        # 初始化子划分器列表，包含一个初始的SubdivideMeshes对象
        subdivider_list = [SubdivideMeshes(mesh)]
        # 根据指定的子划分次数进行循环
        for i in range(subdivide_num-1):
            # 使用列表中的最后一个子划分器对网格进行子划分
            mesh = subdivider_list[-1](mesh)
            # 将新的子划分器添加到列表中
            subdivider_list.append(SubdivideMeshes(mesh))
        # 返回子划分器列表
        return subdivider_list

    # 语义 目前没用
    def assign_semantics_by_nearest(self,new_verts, orig_verts, orig_sem_labels):
    # 计算距离矩阵：(N1, N0)
        dists = torch.cdist(new_verts, orig_verts)  # Euclidean distance
        # 每个新点找最近的旧点索引
        nearest_indices = torch.argmin(dists, dim=1)  # (N1,)
        # 用旧点语义标签赋值
        new_sem_labels = orig_sem_labels[nearest_indices]
        return new_sem_labels
    
    
    
    #对输入的三维网格进行上采样，以增加其细节
    # # def upsample_mesh(self, vert, feat_list=None, sem_labels=None, uplist=None):
        
    #     # print('seg:',sem_labels.shape)
    #     # print('uplist:',uplist.shape)
        
        
    #     face = torch.LongTensor(self.face).to(self.device)
    #     mesh = Meshes(vert[None, :, :], face[None, :, :])

    #     # === 准备特征和语义 ===
    #     if feat_list is not None:
    #         feat_dims = [x.shape[1] for x in feat_list]
    #         # 将所有特征张量拼接成一个大张量
    #         feats = torch.cat(feat_list, 1)

    #     # if sem_labels is not None:
    #     #     sem_labels = torch.from_numpy(sem_labels).to(vert.device)
    #     #     sem_labels = sem_labels.clone()  # 避免 in-place 操作报错

    #     # === 第一阶段：上采样网格 + 特征 + 语义 ===
    #     for subdivider in self.subdivider_list:
    #         if feat_list is not None:
    #             mesh, feats = subdivider(mesh, feats)
    #         else:
    #             mesh = subdivider(mesh)

    #         # 同步语义标签的上采样（每轮复制一遍）
    #     if sem_labels is not None:
    #         print("mesh.verts_list()[0]",mesh.verts_list()[0].shape)
    #         sem_labels = self.assign_semantics_by_nearest(mesh.verts_list()[0],vert,sem_labels)
        

    #     # 更新顶点和特征
    #     vert = mesh.verts_list()[0]

    #     if feat_list is not None:
    #         # print("feats.shape")
    #         # print(feats.shape)#torch.Size([1, 167390, 1548])
    #         feats = feats[0]
            
    #         # saved_feats = feats.clone()  # 保存副本，供第二阶段用
    #         feat_list = torch.split(feats, feat_dims, dim=1)

    #     # from scipy.spatial import cKDTree

    #     # 构建 SMPL-X 表面的 KDTree，只做一次可以缓存
    #     smplx_surface_points = vert.clone().detach().cpu().numpy()  # 第一阶段后的 vert（subdivider 输出）
    #     tree = cKDTree(smplx_surface_points)

    #     # === 第二阶段：对 uplist 指定语义区域加密采样 ===
    #     target_num = 8000

    #     if sem_labels is not None and uplist is not None:
    #         print("sem_labels unique:", torch.unique(sem_labels))
    #         up_mask = torch.isin(sem_labels, uplist)  # [N] bool

    #         fine_points_src = vert[up_mask]
    #         print("可采样点数量：", fine_points_src.shape[0])

    #         if fine_points_src.shape[0] >= target_num:
    #             indices = torch.randperm(fine_points_src.shape[0])[:target_num]
    #             fine_points = fine_points_src[indices]
    #         else:
    #             repeat_times = target_num // fine_points_src.shape[0] + 1
    #             fine_points = fine_points_src.repeat((repeat_times, 1))[:target_num]

    #         # ✅ 添加小扰动
    #         noise = torch.randn_like(fine_points) * 0.005
    #         fine_points_noisy = fine_points + noise

    #         # ✅ 将扰动后的点投影回 SMPL-X 表面（KDTree 最近邻）
    #         _, nearest_idxs = tree.query(fine_points_noisy.detach().cpu().numpy(), k=1)
    #         fine_points_projected = torch.from_numpy(smplx_surface_points[nearest_idxs]).to(vert.device)

    #         # ✅ 拼接到原始点中（防止飘起来）
    #         vert = torch.cat([vert, fine_points_projected], dim=0)

    #         # ✅ 同步特征上采样
    #         if feat_list is not None:
    #             feats = torch.cat(feat_list, dim=1)  # [N, D]
    #             fine_feats_src = feats[up_mask]

    #             if fine_feats_src.shape[0] >= target_num:
    #                 fine_feats = fine_feats_src[indices]
    #             else:
    #                 fine_feats = fine_feats_src.repeat((repeat_times, 1))[:target_num]

    #             # ✅ 替换为投影点对应的特征
    #             fine_feats = feats[nearest_idxs]
    #             feats = torch.cat([feats, fine_feats], dim=0)
    #             feat_list = torch.split(feats, feat_dims, dim=1)

    #         # ✅ 同步语义标签
    #         fine_sem = sem_labels[up_mask]
    #         if fine_sem.shape[0] >= target_num:
    #             fine_sem = fine_sem[indices]
    #         else:
    #             fine_sem = fine_sem.repeat(repeat_times)[:target_num]

    #         fine_sem = sem_labels[nearest_idxs]
    #         sem_labels = torch.cat([sem_labels, fine_sem], dim=0)

    #     # === 返回结构 ===
    #     if feat_list is not None:
    #         return vert, *feat_list, sem_labels
    #     else:
    #         return vert, sem_labels



    
    def upsample_mesh(self, vert, feat_list=None):
        # 将面片索引转换为长整型张量并移动到 GPU 上
        face = torch.LongTensor(self.face).to(self.device)
        # 创建一个 Meshes 对象，包含输入的顶点和面片
        mesh = Meshes(vert[None,:,:], face[None,:,:])
        
        # 如果没有提供特征列表，则只对网格进行上采样
        if feat_list is None:
            # 遍历所有上采样器（subdivider）
            for subdivider in self.subdivider_list:
                # 应用上采样器来增加网格的细节
                mesh = subdivider(mesh)
            # 获取上采样后的顶点列表中的第一个元素
            vert = mesh.verts_list()[0]
            # 返回上采样后的顶点坐标
            return vert
        else:
            # 如果提供了特征列表，则同时对网格和特征进行上采样
            # 计算每个特征张量的维度
            feat_dims = [x.shape[1] for x in feat_list]
            # 将所有特征张量拼接成一个大张量
            feats = torch.cat(feat_list, 1)
            
            # 遍历所有上采样器
            for subdivider in self.subdivider_list:
                # 应用上采样器来增加网格的细节，并同时更新特征张量
                mesh, feats = subdivider(mesh, feats)
            
            # 获取上采样后的顶点列表中的第一个元素
            vert = mesh.verts_list()[0]
            # 获取上采样后的特征张量列表中的第一个元素
            feats = feats[0]
            # 根据原始的特征维度将特征张量分割成多个独立的特征张量
            feat_list = torch.split(feats, feat_dims, dim=1)
            
            # 返回上采样后的顶点坐标和分割后的特征列表
            return vert, *feat_list
            


       
    
    
    
    
    
    #在三维模型中添加空腔，特别是在唇部区域，通常是为了模拟人口腔的内部结构
    def add_cavity(self):
        # 定义唇部顶点的索引列表
        lip_vertex_idx = [2844, 2855, 8977, 1740, 1730, 1789, 8953, 2892]
        
        # 创建一个与顶点数量相同的零向量，用于标记空腔所在的顶点
        is_cavity = np.zeros((self.vertex_num), dtype=np.float32)
        # 将唇部顶点标记为空腔的一部分
        is_cavity[lip_vertex_idx] = 1.0

        # 定义空腔的面片索引列表
        cavity_face = [[0,1,7], [1,2,7], [2, 3,5], [3,4,5], [2,5,6], [2,6,7]]
        
        # 复制原始的面片索引列表
        face_new = list(self.face_orig)
        # 遍历空腔的面片索引列表
        for face in cavity_face:
            v1, v2, v3 = face
            # 将空腔的面片添加到新的面片列表中
            face_new.append([lip_vertex_idx[v1], lip_vertex_idx[v2], lip_vertex_idx[v3]])
        
        # 将新的面片列表转换为整数类型 numpy 数组
        face_new = np.array(face_new, dtype=np.int64)
        
        # 返回空腔顶点标记和新的面片索引数组
        return is_cavity, face_new
#  获取与表情参数相关的顶点索引，排除颈部和眼球区域的顶点，用于面部建模和动画中，以确定哪些顶点会受到表情参数的影响
    def get_expr_vertex_idx(self):
        # FLAME 2020 has all vertices of expr_vertex_idx. use FLAME 2019# 从指定路径加载 FLAME 2019 模型数据
        with open(osp.join(cfg.human_model_path, 'flame', '2019', 'generic_model.pkl'), 'rb') as f:
            flame_2019 = pickle.load(f, encoding='latin1')
        # 找到所有与表情参数相关的顶点索引
        vertex_idxs = np.where((flame_2019['shapedirs'][:,:,300:300+self.expr_param_dim] != 0).sum((1,2)) > 0)[0] # FLAME.SHAPE_SPACE_DIM == 300

        # exclude neck and eyeball regions# # 排除颈部和眼球区域的顶点
        flame_joints_name = ('Neck', 'Head', 'Jaw', 'L_Eye', 'R_Eye')
        expr_vertex_idx = []
        flame_vertex_num = flame_2019['v_template'].shape[0]
        is_neck_eye = torch.zeros((flame_vertex_num)).float()
        # 标记颈部和眼球区域的顶点
        is_neck_eye[flame_2019['weights'].argmax(1)==flame_joints_name.index('Neck')] = 1
        is_neck_eye[flame_2019['weights'].argmax(1)==flame_joints_name.index('L_Eye')] = 1
        is_neck_eye[flame_2019['weights'].argmax(1)==flame_joints_name.index('R_Eye')] = 1
        # 遍历所有与表情参数相关的顶点索引
        for idx in vertex_idxs:
            if is_neck_eye[idx]:
                continue
            expr_vertex_idx.append(idx)
        # 将顶点索引转换为 numpy 数组，并根据 face_vertex_idx 进行索引
        expr_vertex_idx = np.array(expr_vertex_idx)
        expr_vertex_idx = self.face_vertex_idx[expr_vertex_idx]
        # 返回表情相关的顶点索引
        return expr_vertex_idx

smpl_x = SMPLX()
