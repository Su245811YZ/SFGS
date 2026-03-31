import numpy as np
import torch
import torch.nn.functional as F
import os.path as osp
from config import cfg
import smplx
from config import cfg



class FLAME(object):
    def __init__(self):
        # 初始化FLAME模型的参数维度
        self.shape_param_dim = 100  # 形状参数的维度
        self.expr_param_dim = 50    # 表情参数的维度
        self.device = torch.device('cuda')
        # 设置FLAME层的参数
        self.layer_arg = {
            'create_betas': False,  # 不创建形状参数
            'create_expression': False,  # 不创建表情参数
            'create_global_orient': False,  # 不创建全局方向参数
            'create_neck_pose': False,  # 不创建颈部姿态参数
            'create_jaw_pose': False,  # 不创建颌部姿态参数
            'create_leye_pose': False,  # 不创建左眼姿态参数
            'create_reye_pose': False,  # 不创建右眼姿态参数
            'create_transl': False  # 不创建平移参数
        }

        # 创建FLAME层
        self.layer = smplx.create(
            cfg.human_model_path,  # FLAME模型的路径
            'flame',  # 模型类型
            gender='neutral',  # 性别
            num_betas=self.shape_param_dim,  # 形状参数的数量
            num_expression_coeffs=self.expr_param_dim,  # 表情参数的数量
            use_face_contour=True,  # 是否使用面部轮廓
            **self.layer_arg  # 其他层参数
        )

        # 设置顶点数量和面部索引
        self.vertex_num = 5023
        self.face = self.layer.faces.astype(np.int64)  # 面部索引转换为64位整数

        # 加载纹理模型
        self.vertex_uv, self.face_uv = self.load_texture_model()

    def load_texture_model(self):
        # 加载纹理模型文件
        texture = np.load(osp.join(cfg.human_model_path, 'flame', 'FLAME_texture.npz'))
        vertex_uv, face_uv = texture['vt'], texture['ft'].astype(np.int64)
        # 纹理坐标Y轴翻转
        vertex_uv[:,1] = 1 - vertex_uv[:,1]
        return vertex_uv, face_uv

    def set_texture(self, texture, texture_mask):
        # 设置纹理和纹理掩码
        self.texture = texture.to(self.device)  # 将纹理移动到CUDA设备
        self.texture_mask = texture_mask.to(self.device)  # 将纹理掩码移动到CUDA设备

# 实例化FLAME类
flame = FLAME()


