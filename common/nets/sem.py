import math  # 导入数学库，进行数值运算
import numpy as np  # 导入 NumPy 库，进行数组和矩阵运算
import torch  # 导入 PyTorch 库，进行深度学习模型操作
from scipy.spatial.transform import Rotation  # 导入 scipy 库中的 Rotation 模块，用于旋转矩阵的计算
# from scene.gaussian_model import BasicPointCloud  # 从自定义模块导入 BasicPointCloud 类，表示点云数据
from plyfile import PlyData, PlyElement  # 从 plyfile 库导入读取和写入 ply 文件的功能
import open3d as o3d  # 导入 open3d 库，用于 3D 点云处理和可视化
from torch import optim  # 导入 PyTorch 中的优化器模块
import pandas as pd  # 导入 pandas 库，用于数据处理和分析
import torch.nn as nn  # 导入 PyTorch 中的神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 中的函数式 API

# 定义一个函数，用于获取 SMPL 模型中的关节（骨骼）变换矩阵
def get_02v_bone_transforms(Jtr):
    # 定义两个旋转矩阵，分别用于顺时针和逆时针旋转 45 度
    rot45p = Rotation.from_euler('z', 45, degrees=True).as_matrix()
    rot45n = Rotation.from_euler('z', -45, degrees=True).as_matrix()

    # 初始化一个骨骼变换矩阵（4x4），所有骨骼变换默认为单位矩阵
    bone_transforms_02v = np.tile(np.eye(4), (24, 1, 1))

    # 第一条链：左侧髋部（1），左侧膝盖（4），左侧脚踝（7），左侧脚（10）
    chain = [1, 4, 7, 10]
    rot = rot45p.copy()  # 旋转矩阵为顺时针 45 度
    for i, j_idx in enumerate(chain):
        bone_transforms_02v[j_idx, :3, :3] = rot  # 设置旋转矩阵
        t = Jtr[j_idx].copy()  # 获取关节位置
        if i > 0:  # 如果是后续骨骼，考虑父骨骼的位置
            parent = chain[i - 1]
            t_p = Jtr[parent].copy()
            t = np.dot(rot, t - t_p)  # 计算相对位置
            t += bone_transforms_02v[parent, :3, -1].copy()  # 累加父骨骼的变换

        bone_transforms_02v[j_idx, :3, -1] = t  # 设置平移矩阵

    bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)  # 修正平移误差

    # 第二条链：右侧髋部（2），右侧膝盖（5），右侧脚踝（8），右侧脚（11）
    chain = [2, 5, 8, 11]
    rot = rot45n.copy()  # 旋转矩阵为逆时针 45 度
    for i, j_idx in enumerate(chain):
        bone_transforms_02v[j_idx, :3, :3] = rot  # 设置旋转矩阵
        t = Jtr[j_idx].copy()  # 获取关节位置
        if i > 0:
            parent = chain[i - 1]
            t_p = Jtr[parent].copy()
            t = np.dot(rot, t - t_p)  # 计算相对位置
            t += bone_transforms_02v[parent, :3, -1].copy()  # 累加父骨骼的变换

        bone_transforms_02v[j_idx, :3, -1] = t  # 设置平移矩阵

    bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)  # 修正平移误差

    return bone_transforms_02v  # 返回最终的骨骼变换矩阵

# 定义函数，用于读取 ply 文件并将其转换为 BasicPointCloud 对象
# def fetchPly(path):
#     plydata = PlyData.read(path)  # 读取 ply 文件
#     vertices = plydata['vertex']  # 提取顶点数据
#     positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T  # 获取点云坐标
#     colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0  # 获取点云颜色
#     normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T  # 获取点云法线
#     return BasicPointCloud(points=positions, colors=colors, normals=normals)  # 返回点云对象

# 定义函数，用于保存点云数据到 ply 文件
def storePly(path, xyz, rgb):
    # 定义 ply 文件中顶点的 dtype
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)  # 初始化法线为零

    elements = np.empty(xyz.shape[0], dtype=dtype)  # 创建空的元素数组
    attributes = np.concatenate((xyz, normals, rgb), axis=1)  # 合并坐标、法线和颜色信息
    elements[:] = list(map(tuple, attributes))  # 将数据转换为元组并赋值

    # 创建 PlyElement 对象并写入文件
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)  # 写入 ply 文件

# 定义 AABB 类，用于进行空间坐标归一化、反归一化等操作
class AABB(torch.nn.Module):
    def __init__(self, coord_max, coord_min):
        super().__init__()
        self.register_buffer("coord_max", torch.from_numpy(coord_max).float())  # 最大坐标
        self.register_buffer("coord_min", torch.from_numpy(coord_min).float())  # 最小坐标

    # 坐标归一化操作
    def normalize(self, x, sym=False):
        x = (x - self.coord_min) / (self.coord_max - self.coord_min)  # 归一化
        if sym:
            x = 2 * x - 1.  # 如果对称，调整范围到 [-1, 1]
        return x

    # 坐标反归一化操作
    def unnormalize(self, x, sym=False):
        if sym:
            x = 0.5 * (x + 1)  # 对称时，将范围调整到 [0, 1]
        x = x * (self.coord_max - self.coord_min) + self.coord_min  # 反归一化
        return x

    # 限制坐标值在 [coord_min, coord_max] 范围内
    def clip(self, x):
        return x.clip(min=self.coord_min, max=self.coord_max)

    # 返回空间的体积尺度
    def volume_scale(self):
        return self.coord_max - self.coord_min

    # 计算空间的尺度（假设空间是均匀的）
    def scale(self):
        return math.sqrt((self.volume_scale() ** 2).sum() / 3.)  # 求均匀尺度

# 定义一组颜色映射，用于实例分割
COLOR_MAP_INSTANCES = {
    0: (226., 226., 226.),  # 灰色
    1: (120., 94., 240.),   # 紫色
    2: (254., 97., 0.),     # 橙色
    3: (255., 176., 0.),    # 黄色
    4: (100., 143., 255.),  # 蓝色
    5: (220., 38., 127.),   # 粉色
    6: (0., 255., 255.),    # 青色
    7: (255., 204., 153.),  # 浅橙色
    8: (255., 102., 0.),    # 深橙色
    9: (0., 128., 128.),    # 蓝绿色
    10: (153., 153., 255.), # 淡蓝色
}

# 定义一组颜色映射，用于合并的身体部位分割
MERGED_BODY_PART_COLORS = {
    0:  (226., 226., 226.),     # background or undefined
    1:  (158.0, 143.0, 20.0),   # rightHand
    2:  (243.0, 115.0, 68.0),   # rightUpLeg
    3:  (228.0, 162.0, 227.0),  # leftArm
    4:  (210.0, 78.0, 142.0),   # head
    5:  (152.0, 78.0, 163.0),   # leftLeg
    6:  (76.0, 134.0, 26.0),    # leftFoot
    7:  (100.0, 143.0, 255.0),  # torso
    8:  (129.0, 0.0, 50.0),     # rightFoot
    9:  (255., 176., 0.),       # rightArm
    10: (192.0, 100.0, 119.0),  # leftHand
    11: (149.0, 192.0, 228.0),  # rightLeg
    12: (243.0, 232.0, 88.0),   # leftForeArm
    13: (90., 64., 210.),       # rightForeArm
    14: (152.0, 200.0, 156.0),  # leftUpLeg
    15: (129.0, 103.0, 106.0),  # hips
}

class HumanSegmentationDataset():
    def __init__(self, file_list):
        self.file_list = file_list                         # 文件路径列表
        self.ORIG_BODY_PART_IDS = set(range(100, 126))     # 原始人体部位ID集合，用于过滤或映射

    def __len__(self):
        return len(self.file_list)                         # 返回数据集中点云文件数量

    def read_plyfile(self, file_path):
        """读取ply文件为numpy数组，如果文件为空则返回None。"""
        with open(file_path, 'rb') as f:
            plydata = PlyData.read(f)                      # 使用PlyData读取文件内容
        if plydata.elements:                               # 如果有元素数据
            return pd.DataFrame(plydata.elements[0].data).values  # 转为numpy数组返回

    def load_pc(self, file_path):
        """加载点云文件，返回坐标、颜色和原始分割标签。"""
        pc = self.read_plyfile(file_path)                  # 读取点云文件，形状(num_points, 8)

        pc_coords = pc[:, 0:3]                              # 点的三维坐标 (num_points, 3)
        pc_rgb = pc[:, 3:6].astype(np.uint8)               # RGB颜色信息 (0-255)
        pc_orig_segm_labels = pc[:, 6].astype(np.uint8)    # 原始语义标签 (num_points,)
        return pc_coords, pc_rgb, torch.tensor(pc_orig_segm_labels).cuda()  # 转为CUDA张量返回

    def export_colored_pcd_inst_segm(self, coords, pc_inst_labels, write_path):
        """导出点云并用实例分割标签上色保存为文件。"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)    # 设置点坐标
        inst_colors = np.asarray([self.COLOR_MAP_INSTANCES[int(label_idx)] for label_idx in pc_inst_labels]) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(inst_colors)  # 设置颜色
        pcd.estimate_normals()                             # 估算法向量（可选）
        o3d.io.write_point_cloud(write_path, pcd)          # 写入点云文件

    def export_colored_pcd_part_segm(self, coords, pc_part_segm_labels, write_path):
        """导出点云并用部位语义标签上色保存为文件。"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)    # 设置点坐标
        part_colors = np.asarray(
            [MERGED_BODY_PART_COLORS[int(label_idx)] for label_idx in pc_part_segm_labels]) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(part_colors)  # 设置颜色
        pcd.estimate_normals()                             # 估算法向量（可选）
        o3d.io.write_point_cloud(write_path, pcd)          # 写入点云文件

    def __getitem__(self, index):
        """按索引读取一个样本的点云数据。"""
        return self.load_pc(self.file_list[index])
    
    def interpolate_semantic_labels_nn(self,verts_orig, verts_upsampled, segm_labels_orig):
        """
        使用最近邻插值将原始10475个SMPL-X顶点的语义标签扩展到上采样后的所有点
        输入:
            verts_orig: Tensor [10475, 3]
            verts_upsampled: Tensor [N, 3]
            segm_labels_orig: Tensor [10475]
        输出:
            segm_labels_upsampled: Tensor [N]
        """
        with torch.no_grad():
            verts_orig = torch.from_numpy(verts_orig).float().to(verts_upsampled.device)

            dists = torch.cdist(verts_upsampled, verts_orig)  # [N, 10475]
            nn_idx = torch.argmin(dists, dim=1)               # 最近邻索引 [N]
            segm_labels_upsampled = segm_labels_orig[nn_idx]  # 标签传播 [N]
        return segm_labels_upsampled



# 替换成你自己的点云文件路径
ply_file = "/home/suyuze/workspace/ExAvatar_RELEASE/data/smpl_semantic_sim.ply"
new_ply_file = "ExAvatar_RELEASE/data/XHumans/data/00028/train/Take2/SMPLX/mesh-f00090_smplx.ply"
output_file = "newone.ply"

# 加载数据集（只有一个点云文件）
datasetsem = HumanSegmentationDataset([ply_file])

# 获取第一个（也是唯一一个）点云的数据
coords, rgb, segm_labels = datasetsem[0]
croodsnew = dataset.read_plyfile(new_ply_file)
print(segm_labels[0])
    # 将语义标签分割结果保存为彩色点云
    # dataset.export_colored_pcd_part_segm(croodsnew, segm_labels.cpu().numpy(), output_file)
    # # print(coords.shape)
    # # print(rgb)
    # print(coords.shape)
    # print(f"导出完成：{output_file}")
    # print("你可以用 open3d 或 MeshLab 打开它查看颜色标注结果")