import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer
from pytorch3d.renderer import TexturesUV
from config import cfg

# 创建一系列全连接层
def make_linear_layers(feat_dims, relu_final=True, use_gn=False):

    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            if use_gn:
                layers.append(nn.GroupNorm(4, feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def get_face_index_map_xy(mesh, face, cam_param, render_shape):
    batch_size = mesh.shape[0]
    face = torch.from_numpy(face).cuda()[None,:,:].repeat(batch_size,1,1)
    mesh = torch.stack((-mesh[:,:,0], -mesh[:,:,1], mesh[:,:,2]),2) # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(mesh, face)

    cameras = PerspectiveCameras(focal_length=cam_param['focal'],
                                principal_point=cam_param['princpt'],
                                device='cuda',
                                in_ndc=False,
                                image_size=torch.LongTensor(render_shape).cuda().view(1,2))
    raster_settings = RasterizationSettings(image_size=render_shape, blur_radius=0.0, faces_per_pixel=1)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()
    outputs = rasterizer(mesh)
    return outputs

class MeshRenderer(nn.Module):
    
    
    def __init__(self, vertex_uv, face_uv):
        # 初始化方法，接收网格的顶点 UV 和面 UV。
        super(MeshRenderer, self).__init__()  # 调用父类的初始化方法。
        self.vertex_uv = torch.FloatTensor(vertex_uv).cuda()  # 将顶点 UV 转为浮点张量并移动到 CUDA 设备。
        self.face_uv = torch.LongTensor(face_uv).cuda()  # 将面 UV 转为长整型张量并移动到 CUDA 设备。
    '''(
                face_texture, 
                human_asset['mean_3d'][None, smpl_x.face_vertex_idx, :],
                flame.face,
                {k: v[i, None] for k, v in data['cam_param'].items()},
                (img_height, img_width)
            )'''
    def forward(self, uvmap, mesh, face, cam_param, render_shape):
        # 定义前向传播方法，接收 UV 图、网格、面、相机参数和渲染形状。
        batch_size, uvmap_dim, uvmap_height, uvmap_width = uvmap.shape  # 获取输入 UV 图的尺寸信息。
        render_height, render_width = render_shape  # 解包渲染形状以获取高度和宽度。
        # 获取可见面从网格中
        mesh = torch.bmm(cam_param['R'], mesh.permute(0,2,1)).permute(0,2,1) + cam_param['t'].view(-1,1,3)  
        # 计算将网格坐标从世界坐标转换到相机坐标，使用相机的旋转矩阵 R 和平移向量 t。
        
        fragments = get_face_index_map_xy(mesh, face, cam_param, (render_height, render_width))
        # 获取在相机坐标系下渲染图像中可见面的对应信息。

        vertex_uv = torch.stack((self.vertex_uv[:,0], 1 - self.vertex_uv[:,1]),1)[None,:,:].repeat(batch_size,1,1)
        # 将顶点 UV 进行 Y 轴翻转以符合 PyTorch3D 的约定，并扩展维度以匹配批量大小。
        
        renderer = TexturesUV(uvmap.permute(0,2,3,1), self.face_uv[None,:,:].repeat(batch_size,1,1), vertex_uv)
        # 创建一个 TexturesUV 对象，输入 UV 图、面 UV 和已翻转的顶点 UV。

        render = renderer.sample_textures(fragments)  # 从纹理中采样渲染结果，返回渲染图像。
        # render 的形状为 (batch_size, render_height, render_width, faces_per_pixel, uvmap_dim)
        
        render = render[:,:,:,0,:].permute(0,3,1,2)  # 保留每个像素的第一个面并调整张量维度，变为 (batch_size, uvmap_dim, render_height, render_width
        # fg mask
        pix_to_face = fragments.pix_to_face  # 获取每个像素对应的面 ID，形状为 (batch_size, render_height, render_width, faces_per_pixel)，无效值: -1
        pix_to_face_xy = pix_to_face[:,:,:,0]  # 获取每个像素对应的第一个面的 ID。注意：这是打包的表示方式。

        # 打包转为解包
        is_valid = (pix_to_face_xy != -1).float()  # 创建一个表示有效面的掩码。
        
        pix_to_face_xy = (pix_to_face_xy - torch.arange(batch_size)[:,None,None].cuda() * face.shape[0]) * is_valid + (-1) * (1 - is_valid)
        # 将打包的面 ID 转换为解包格式，若无效则设置为 -1。

        pix_to_face_xy = pix_to_face_xy.long()  # 转换为长整型张量。
        
        # 将背景像素设置为 -1
        render[pix_to_face_xy[:,None,:,:].repeat(1,uvmap_dim,1,1) == -1] = -1
        # 在输出渲染中，将对应背景面（无效的像素）设置为 -1。

        return render  # 返回最终的渲染结果。



