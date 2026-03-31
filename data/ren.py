import smplx
import torch
import numpy as np
import matplotlib.pyplot as plt

# ===== 1. 加载 SMPL-X 模型与参数文件路径 =====
model_path = '/home/suyuze/workspace/ExAvatar_mine/common/utils/human_model_files'  # 替换为你本地的SMPL-X模型路径
smplx_file = '/home/suyuze/workspace/ExAvatar_mine/data/XHumans/data/00028/train/Take2/SMPLX/mesh-f00001_smplx.pkl'  # 替换为你的SMPLX参数文件

# ========== 加载 SMPL-X 参数 ========== 
params = np.load(smplx_file, allow_pickle=True)

def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)

device = torch.device('cpu')
body_model = smplx.create(model_path=model_path, model_type='smplx',
                          gender='neutral', ext='npz', use_pca=False,
                          create_global_orient=True,
                          create_body_pose=True,
                          create_betas=True,
                          create_left_hand_pose=True,
                          create_right_hand_pose=True,
                          create_expression=True,
                          create_jaw_pose=True,
                          create_leye_pose=True,
                          create_reye_pose=True).to(device)

output = body_model(
    betas=to_tensor(params['betas']),
    body_pose=to_tensor(params['body_pose']),
    global_orient=to_tensor(params['global_orient']),
    left_hand_pose=to_tensor(params['left_hand_pose']),
    right_hand_pose=to_tensor(params['right_hand_pose']),
    jaw_pose=to_tensor(params['jaw_pose']),
    expression=to_tensor(params['expression']),
    leye_pose=to_tensor(params['leye_pose']),
    reye_pose=to_tensor(params['reye_pose']),
    transl=to_tensor(params.get('transl', np.zeros(3)))
)

vertices = output.vertices[0].detach().cpu().numpy()  # [V, 3]
joints = output.joints[0].detach().cpu().numpy()      # [J, 3]
joints_55 = joints[:55]

# ========== 正视图绘制（X 左右，Y 上下） ========== 
fig, ax = plt.subplots(figsize=(6, 8))

# 画3D点云（灰色，透明）
ax.scatter(vertices[:, 0], vertices[:, 1], c='gray', s=0.1, alpha=0.3)

# 画前55个关节点（红色）
ax.scatter(joints_55[:, 0], joints_55[:, 1], c='red', s=10)

# 去掉背景颜色，设置透明背景
fig.patch.set_facecolor('none')
ax.set_facecolor('none')

# 去掉坐标轴和网格
ax.axis('off')

ax.set_xlabel('X (left-right)')
ax.set_ylabel('Y (up-down)')
ax.set_title('SMPL-X 55 joints and 3D Points (Front View)')

# 保存图像
save_path = 'smplx_joints_and_3d_points_front_view.png'
plt.savefig(save_path, dpi=300, transparent=True)  # 保存时设置透明背景
print(f"图像保存至：{save_path}")
plt.show()
