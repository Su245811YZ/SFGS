import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
import lpips
from utils.smpl_x import smpl_x
from pytorch3d.structures import Meshes



import kornia.color as kc  # 用于颜色空间转换





class BitPlaneLoss(nn.Module):
    def __init__(self, num_bits=8, weighted=False):
        super(BitPlaneLoss, self).__init__()
        self.num_bits = num_bits
        self.weighted = weighted  # 是否给高位更大权重

    def forward(self, img_out, img_target, bbox=None, mask=None, bg=None):
        """
        img_out: [B, C, H, W], 输出图像 (0~1)
        img_target: [B, C, H, W], GT 图像 (0~1)
        """
        batch_size, feat_dim, img_height, img_width = img_out.shape
        
        # mask / bg 处理
        if (mask is not None) and (bg is not None):
            img_target = img_target * mask + (1 - mask) * bg[:, :, None, None]

        # bbox 裁剪
        if bbox is not None:
            xmin, ymin, width, height = [int(x) for x in bbox[0]]
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmin + width, img_width)
            ymax = min(ymin + height, img_height)
            img_out = img_out[:, :, ymin:ymax, xmin:xmax]
            img_target = img_target[:, :, ymin:ymax, xmin:xmax]

        # 转成整数 0~255
        img_out_int = (img_out.clamp(0, 1) * 255).long()
        img_target_int = (img_target.clamp(0, 1) * 255).long()

        losses = []
        for i in range(self.num_bits):
            # 取第 i 个比特平面
            pred_bit = ((img_out_int >> i) & 1).float()
            target_bit = ((img_target_int >> i) & 1).float()
            
            # L1 loss
            bit_loss = torch.abs(pred_bit - target_bit)
            
            # 权重：高位更重要
            if self.weighted:
                w = 2 ** i
                bit_loss = w * bit_loss
            
            losses.append(bit_loss)

        # 平均所有 bit 平面的损失
        loss = sum(losses) / self.num_bits
        return loss



class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, img_out, img_target, bbox=None, mask=None, bg=None):
        batch_size, feat_dim, img_height, img_width = img_out.shape

        # 应用 mask 和背景
        if (mask is not None) and (bg is not None):
            img_target = img_target * mask + (1 - mask) * bg[:, :, None, None]

        # 应用裁剪区域
        if bbox is not None:
            xmin, ymin, width, height = [int(x) for x in bbox[0]]
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmin + width, img_width)
            ymax = min(ymin + height, img_height)
            img_out = img_out[:, :, ymin:ymax, xmin:xmax]
            img_target = img_target[:, :, ymin:ymax, xmin:xmax]

        # 计算梯度（x 和 y 方向）
        grad_out_x = img_out[:, :, :, :-1] - img_out[:, :, :, 1:]
        grad_out_y = img_out[:, :, :-1, :] - img_out[:, :, 1:, :]

        grad_target_x = img_target[:, :, :, :-1] - img_target[:, :, :, 1:]
        grad_target_y = img_target[:, :, :-1, :] - img_target[:, :, 1:, :]

        loss = (grad_out_x - grad_target_x).abs().mean() + (grad_out_y - grad_target_y).abs().mean()
        return loss


class LabLoss(nn.Module):
    def __init__(self):
        super(LabLoss, self).__init__()

    def forward(self, img_out, img_target, bbox=None, mask=None, bg=None):
        batch_size, feat_dim, img_height, img_width = img_out.shape
        assert feat_dim == 3, "Input must be RGB with 3 channels."

        # 使用 mask + bg 融合背景
        if (mask is not None) and (bg is not None):
            img_target = img_target * mask + (1 - mask) * bg[:, :, None, None]

        # 使用 bbox 裁剪
        if bbox is not None:
            xmin, ymin, width, height = [int(x) for x in bbox[0]]
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmin + width, img_width)
            ymax = min(ymin + height, img_height)
            img_out = img_out[:, :, ymin:ymax, xmin:xmax]
            img_target = img_target[:, :, ymin:ymax, xmin:xmax]

        # RGB → Lab（值域需在 [0,1]）
        img_out_lab = kc.rgb_to_lab(img_out.clamp(0, 1))
        img_target_lab = kc.rgb_to_lab(img_target.clamp(0, 1))

        # 计算 Lab 空间的 L1 损失
        loss = torch.abs(img_out_lab - img_target_lab)
        return loss




class RGBLoss(nn.Module):
    def __init__(self):
        super(RGBLoss, self).__init__()
    
    def forward(self, img_out, img_target, bbox=None, mask=None, bg=None):
        
        batch_size, feat_dim, img_height, img_width = img_out.shape
        if (mask is not None) and (bg is not None):
            img_target = img_target * mask + (1 - mask) * bg[:,:,None,None]
        if bbox is not None:
            xmin, ymin, width, height = [int(x) for x in bbox[0]]
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmin+width, img_width)
            ymax = min(ymin+height, img_height)
            img_out = img_out[:,:,ymin:ymax,xmin:xmax]
            img_target = img_target[:,:,ymin:ymax,xmin:xmax]

        loss = torch.abs(img_out - img_target)
        return loss

class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()

    def gaussian(self, window_size, sigma):
        gauss = torch.FloatTensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]).cuda()
        return gauss / gauss.sum()

    def create_window(self, window_size, feat_dim):
        window_1d = self.gaussian(window_size, 1.5)[:,None]
        window_2d = torch.mm(window_1d, window_1d.permute(1,0))[None,None,:,:]
        window_2d = window_2d.repeat(feat_dim,1,1,1)
        return window_2d

    def forward(self, img_out, img_target, bbox=None, mask=None, bg=None, window_size=11):
        batch_size, feat_dim, img_height, img_width = img_out.shape
        if (mask is not None) and (bg is not None):
            img_target = img_target * mask + (1 - mask) * bg[:,:,None,None]
        if bbox is not None:
            xmin, ymin, width, height = [int(x) for x in bbox[0]]
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmin+width, img_width)
            ymax = min(ymin+height, img_height)
            img_out = img_out[:,:,ymin:ymax,xmin:xmax]
            img_target = img_target[:,:,ymin:ymax,xmin:xmax]

        window = self.create_window(window_size, feat_dim)
        mu1 = F.conv2d(img_out, window, padding=window_size//2, groups=feat_dim)
        mu2 = F.conv2d(img_target, window, padding=window_size//2, groups=feat_dim)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img_out*img_out, window, padding=window_size//2, groups=feat_dim) - mu1_sq
        sigma2_sq = F.conv2d(img_target*img_target, window, padding=window_size//2, groups=feat_dim) - mu2_sq
        sigma1_sigma2 = F.conv2d(img_out*img_target, window, padding=window_size//2, groups=feat_dim) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma1_sigma2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map

# image perceptual loss (LPIPS. https://github.com/richzhang/PerceptualSimilarity)
class LPIPS(nn.Module):
    def __init__(self):
        super(LPIPS, self).__init__()
        # 初始化 LPIPS 模型，使用 VGG 网络作为特征提取器，并将模型移动到 GPU 上
        self.lpips = lpips.LPIPS(net='vgg').cuda()

    def forward(self, img_out, img_target, bbox=None, mask=None, bg=None):
        # 获取输入图像的尺寸信息
        batch_size, feat_dim, img_height, img_width = img_out.shape
        
        # 如果提供了掩码和背景，则对目标图像进行调整
        if (mask is not None) and (bg is not None):
            img_target = img_target * mask + (1 - mask) * bg[:,:,None,None]
        
        # 如果提供了边界框，则对图像进行裁剪
        if bbox is not None:
            xmin, ymin, width, height = [int(x) for x in bbox[0]]
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmin+width, img_width)
            ymax = min(ymin+height, img_height)
            img_out = img_out[:,:,ymin:ymax,xmin:xmax]
            img_target = img_target[:,:,ymin:ymax,xmin:xmax]
        
        # 将图像从 [0, 1] 范围转换到 [-1, 1] 范围，以适应 LPIPS 模型的输入要求
        img_out = img_out * 2 - 1
        img_target = img_target * 2 - 1
        
        # 计算并返回 LPIPS 损失
        loss = self.lpips(img_out, img_target)
        return loss
# from sklearn.neighbors import NearestNeighbors


# class LaplacianReg(nn.Module):
#     def __init__(self, vertex_pos, face=None, neighbor_max_num=10, use_knn=True):
#         """
#         vertex_pos: Tensor or numpy, shape [N, 3]
#         face: numpy array of shape [F, 3] if use_knn=False
#         use_knn: whether to build neighbors using KNN instead of face
#         """
#         super(LaplacianReg, self).__init__()
#         if isinstance(vertex_pos, torch.Tensor):
#             vertex_pos = vertex_pos.detach().cpu().numpy()

#         if use_knn:
#             self.neighbor_idxs, self.neighbor_weights = self.get_knn_neighbor(vertex_pos, neighbor_max_num)
#         else:
#             assert face is not None, "Face must be provided if not using KNN"
#             self.neighbor_idxs, self.neighbor_weights = self.get_face_neighbor(len(vertex_pos), face, neighbor_max_num)

#     def get_face_neighbor(self, vertex_num, face, neighbor_max_num=10):
#         # 基于三角面构建邻接
#         adj = {i: set() for i in range(vertex_num)}
#         for tri in face:
#             for i in tri:
#                 adj[i] |= set(tri) - {i}

#         neighbor_idxs = np.tile(np.arange(vertex_num)[:, None], (1, neighbor_max_num))
#         neighbor_weights = np.zeros((vertex_num, neighbor_max_num), dtype=np.float32)

#         for idx in range(vertex_num):
#             neighbors = list(adj[idx])
#             neighbor_num = min(len(neighbors), neighbor_max_num)
#             if neighbor_num > 0:
#                 neighbor_idxs[idx, :neighbor_num] = np.array(neighbors)[:neighbor_num]
#                 neighbor_weights[idx, :neighbor_num] = -1.0 / neighbor_num

#         return torch.from_numpy(neighbor_idxs).long().cuda(), torch.from_numpy(neighbor_weights).float().cuda()

#     def get_knn_neighbor(self, verts, k=10):
#         # 基于KNN构建邻居
#         nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(verts)
#         _, indices = nbrs.kneighbors(verts)
#         neighbor_idxs = indices[:, 1:]  # 去除自己
#         neighbor_weights = -1.0 / k * np.ones_like(neighbor_idxs, dtype=np.float32)

#         return torch.from_numpy(neighbor_idxs).long().cuda(), torch.from_numpy(neighbor_weights).float().cuda()

#     def compute_laplacian(self, x, neighbor_idxs, neighbor_weights):
#         """
#         x: [B, N, 3]
#         neighbor_idxs: [N, K]
#         neighbor_weights: [N, K]
#         """
#         B, N, _ = x.shape
#         neighbors = x[:, neighbor_idxs]  # [B, N, K, 3]
#         lap = x + (neighbors * neighbor_weights[None, :, :, None]).sum(2)  # [B, N, 3]
#         return lap

#     def forward(self, out, target=None):
#         """
#         out: [B, N, 3]
#         target: [B, N, 3] or None
#         """
#         lap_out = self.compute_laplacian(out, self.neighbor_idxs, self.neighbor_weights)
#         if target is None:
#             loss = (lap_out ** 2).mean()
#         else:
#             lap_target = self.compute_laplacian(target, self.neighbor_idxs, self.neighbor_weights)
#             loss = ((lap_out - lap_target) ** 2).mean()
#         return loss


# “输出的 mesh 的 Laplacian 形状”和“目标 mesh 的 Laplacian 形状”之间的差异平方”
class LaplacianReg(nn.Module):
    def __init__(self, vertex_num, face):
        super(LaplacianReg, self).__init__()
        # 初始化时计算每个顶点的邻居索引和邻居权重
        self.neighbor_idxs, self.neighbor_weights = self.get_neighbor(vertex_num, face)

    def get_neighbor(self, vertex_num, face, neighbor_max_num=10):
        # 构建邻接表，记录每个顶点的邻居顶点
        adj = {i: set() for i in range(vertex_num)}
        for i in range(len(face)):
            for idx in face[i]:
                adj[idx] |= set(face[i]) - set([idx])

        # 初始化邻居索引和邻居权重矩阵
        neighbor_idxs = np.tile(np.arange(vertex_num)[:, None], (1, neighbor_max_num))
        neighbor_weights = np.zeros((vertex_num, neighbor_max_num), dtype=np.float32)
        for idx in range(vertex_num):
            # 计算每个顶点的邻居数量，并更新邻居索引和权重
            neighbor_num = min(len(adj[idx]), neighbor_max_num)
            neighbor_idxs[idx, :neighbor_num] = np.array(list(adj[idx]))[:neighbor_num]
            neighbor_weights[idx, :neighbor_num] = -1.0 / neighbor_num
        
        # 将邻居索引和权重转换为CUDA张量
        neighbor_idxs, neighbor_weights = torch.from_numpy(neighbor_idxs).cuda(), torch.from_numpy(neighbor_weights).cuda()
        return neighbor_idxs, neighbor_weights
    
    def compute_laplacian(self, x, neighbor_idxs, neighbor_weights):
        # 计算拉普拉斯平滑
        lap = x + (x[:, neighbor_idxs] * neighbor_weights[None, :, :, None]).sum(2)
        return lap

    def forward(self, out, target):
        # 前向传播，计算损失
        if target is None:
            # 如果没有目标顶点，只计算输出顶点的拉普拉斯平滑
            lap_out = self.compute_laplacian(out, self.neighbor_idxs, self.neighbor_weights)
            loss = lap_out ** 2
            return loss
        else:
            # 如果有目标顶点，计算输出顶点和目标顶点的拉普拉斯平滑，并计算它们之间的差异
            lap_out = self.compute_laplacian(out, self.neighbor_idxs, self.neighbor_weights)
            lap_target = self.compute_laplacian(target, self.neighbor_idxs, self.neighbor_weights)
            loss = (lap_out - lap_target) ** 2
            return loss

class JointOffsetSymmetricReg(nn.Module):
    def __init__(self):
        super(JointOffsetSymmetricReg, self).__init__()
    
    def forward(self, joint_offset):
        right_joint_idx, left_joint_idx = [], []
        for j in range(smpl_x.joint_num):
            if smpl_x.joints_name[j][:2] == 'R_':
                right_joint_idx.append(j)
                idx = smpl_x.joints_name.index('L_' + smpl_x.joints_name[j][2:])
                left_joint_idx.append(idx)

        loss = torch.abs(joint_offset[right_joint_idx,0] + joint_offset[left_joint_idx,0]) + torch.abs(joint_offset[right_joint_idx,1] - joint_offset[left_joint_idx,1]) + torch.abs(joint_offset[right_joint_idx,2] - joint_offset[left_joint_idx,2])
        return loss
#这个 HandMeanReg 类实现了一个正则化损失函数，用于手部姿态估计任务。
# 它通过计算中立姿态手部网格的法线向量与偏移量之间的点积，并确保点积结果非负，来鼓励手部姿态的合理性和稳定性。
class HandMeanReg(nn.Module):
    def __init__(self):
        super(HandMeanReg, self).__init__()
 
    def forward(self, mesh_neutral_pose, offset, is_rhand, is_lhand):
        batch_size = offset.shape[0]
        is_hand = (is_rhand + is_lhand) > 0
        with torch.no_grad():
            normal = Meshes(verts=mesh_neutral_pose[None,:,:], faces=torch.LongTensor(smpl_x.face_upsampled).cuda()[None,:,:]).verts_normals_packed().reshape(1,smpl_x.vertex_num_upsampled,3).detach().repeat(batch_size,1,1)
        dot_prod = torch.sum(normal * F.normalize(offset, p=2, dim=2), 2)[:,is_hand]#余弦值，方向一致性
        loss = torch.clamp(dot_prod, min=0)
        return loss

import torch
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_laplacian_smoothing

# def minimal_surface_loss(mesh: Meshes):
#     # 面积损失
#     faces = mesh.faces_packed()
#     verts = mesh.verts_packed()
#     areas = compute_triangle_areas(faces, verts)
#     area_loss = areas.sum()

#     # 曲率损失
#     curvature_loss = mesh_laplacian_smoothing(mesh)
    
#     # 总损失
#     alpha, beta = 1.0, 0.1
#     total_loss = alpha * area_loss + beta * curvature_loss
#     return total_loss

class MinimalSurfaceLoss(nn.Module):
    def __init__(self,alpha=1.0,beta=0.1):
        super(MinimalSurfaceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        
    def compute_triangle_areas(self,faces, verts):
        # faces: [num_faces, 3]  -  每个面由三个顶点索引组成
        # verts: [num_verts, 3]  -  每个顶点的坐标
        
        v0 = verts[faces[..., 0], :]
        v1 = verts[faces[..., 1], :]
        v2 = verts[faces[..., 2], :]
        # 使用叉积计算三角形面积
        face_areas = 0.5 * torch.cross(v1 - v0, v2 - v0).norm(dim=1)
        return face_areas
    
    def compute_area_loss(self,faces,verts):
        # faces =mesh.faces_packed()
        # verts = mesh.verts_packed()
        areas = self.compute_triangle_areas(faces,verts)
        return areas.sum()
    
    def compute_curvature_loss(self, mesh):
        return mesh_laplacian_smoothing(mesh)
    
    def forward(self, verts):
        faces=torch.LongTensor(smpl_x.face_upsampled).cuda()[None,:,:]
        mesh=Meshes(verts=verts[None,:,:], faces=torch.LongTensor(smpl_x.face_upsampled).cuda()[None,:,:])
        total_loss = self.alpha * self.compute_area_loss(faces,verts) + self.beta * self.compute_curvature_loss(mesh)
        return total_loss
    
    
from pytorch3d.ops import knn_points

class SemanticConsistencyLoss(nn.Module):
    def __init__(self, k=1):
        super(SemanticConsistencyLoss, self).__init__()
        self.k = k

    def forward(self, mean_3d, mean_3d_cano, segm_labels):
        """
        mean_3d: [N, 3] posed 高斯点
        mean_3d_cano: [N, 3] canonical 高斯点
        segm_labels: [N] canonical 语义标签
        """
        # 需要 [B, N, 3] 的输入格式，这里 B=1
        knn = knn_points(mean_3d[None], mean_3d_cano[None], K=self.k, return_nn=False)
        nearest_ids = knn.idx[0, :, 0]  # [N]表示每个 posed 点最近的 1 个 canonical 点的 索引
        
        matched_labels = segm_labels[nearest_ids]  # 最近 canonical 点的标签
        semantic_inconsistent = (matched_labels != segm_labels)  # 和原始标签对比（约等于 ground truth）
        semantic_loss = semantic_inconsistent.float().mean()

        return semantic_loss


# #伪监督，不需要GT
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import cv2
# import numpy as np

# class FlowConsistencyLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     @staticmethod
#     def compute_opencv_flow(img1, img2):
#         """
#         使用 OpenCV Farneback 方法计算光流（灰度图）
#         输入 img1, img2: [B, 3, H, W] 范围 [0,1] 的 tensor
#         返回 flow: [B, 2, H, W] 的 tensor
#         """
#         flows = []
#         for i in range(img1.shape[0]):
#             im1 = img1[i].permute(1, 2, 0).cpu().numpy()
#             im2 = img2[i].permute(1, 2, 0).cpu().numpy()
#             im1_gray = cv2.cvtColor((im1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
#             im2_gray = cv2.cvtColor((im2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

#             flow = cv2.calcOpticalFlowFarneback(im1_gray, im2_gray, None,
#                                                 pyr_scale=0.5, levels=3, winsize=15,
#                                                 iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
#             flow = torch.from_numpy(flow).permute(2, 0, 1)  # [2, H, W]
#             flows.append(flow)

#         return torch.stack(flows, dim=0)  # [B, 2, H, W]

#     def forward(self, prev_img, curr_img):
#         """
#         输入：两帧图像 [B, 3, H, W]，输出 flow 重建 loss
#         """
#         flow = self.compute_opencv_flow(prev_img, curr_img).to(curr_img.device)

#         # 使用光流将 prev_img warp 到 curr_img
#         B, C, H, W = curr_img.shape
#         grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
#         grid = torch.stack((grid_x, grid_y), dim=0).float().to(curr_img.device)  # [2, H, W]
#         grid = grid.unsqueeze(0).repeat(B,1,1,1)  # [B,2,H,W]

#         warped_grid = grid + flow  # [B,2,H,W]

#         # 归一化为 [-1,1]
#         warped_grid[:,0,:,:] = 2.0 * warped_grid[:,0,:,:] / (W - 1) - 1.0
#         warped_grid[:,1,:,:] = 2.0 * warped_grid[:,1,:,:] / (H - 1) - 1.0
#         warped_grid = warped_grid.permute(0, 2, 3, 1)  # [B,H,W,2]

#         warped_prev_img = F.grid_sample(prev_img, warped_grid, mode='bilinear', padding_mode='border')

#         # 与当前图像比对
#         loss = F.l1_loss(warped_prev_img, curr_img)
#         return loss

