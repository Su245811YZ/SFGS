import os
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from tqdm import tqdm
from utils.smpl_x import smpl_x
face = smpl_x.face_vertex_idx
lhand = smpl_x.lhand_vertex_idx
rhand = smpl_x.rhand_vertex_idx




def load_ply_file(file_path):
    mesh = trimesh.load(file_path)
    if hasattr(mesh, 'vertices'):
        return mesh.vertices
    else:
        raise ValueError(f"No vertices found in {file_path}")

def rotate_to_xz_plane(points):
    rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
    return trimesh.transformations.transform_points(points, rotation_matrix)

# def chamfer_distance(points_src, points_tgt):
#     tree_tgt = cKDTree(points_tgt)
#     distances_src, _ = tree_tgt.query(points_src)
#     tree_src = cKDTree(points_src)
#     distances_tgt, _ = tree_src.query(points_tgt)
#     cd = (np.sum(distances_src) + np.sum(distances_tgt)) / (len(distances_src) + len(distances_tgt))
#     cdmax = max(np.max(distances_src), np.max(distances_tgt))
#     return cd, cdmax

# def compute_cd_for_subfolders(base_pred_dir, gt_dir, subfolders):
#     all_cd = []
#     all_cdmax = []

#     for sub in subfolders:
#         pred_folder = os.path.join(base_pred_dir, sub)
#         pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith('.ply')])
        
#         print(f"\n处理子文件夹: {sub}, 共 {len(pred_files)} 个文件")
        
#         for fname in tqdm(pred_files):
#             pred_path = os.path.join(pred_folder, fname)

#             # 自动匹配GT文件名：例如 1_point.ply → Take6_mesh-f00001.ply
#             name_id = fname.replace('_point.ply', '').zfill(5)  # 保证是5位数，如 "1" → "00001"
#             gt_name = f"{sub}_mesh-f{name_id}.ply"
#             gt_path = os.path.join(gt_dir, gt_name)

#             if not os.path.exists(gt_path):
#                 print(f"警告：找不到GT文件 {gt_path}，跳过")
#                 continue

#             try:
#                 pred_points = rotate_to_xz_plane(load_ply_file(pred_path))
#                 gt_points = load_ply_file(gt_path)

#                 cd, cdmax = chamfer_distance(pred_points, gt_points)
#                 all_cd.append(cd)
#                 all_cdmax.append(cdmax)
#             except Exception as e:
#                 print(f"跳过 {fname}，错误：{e}")
    
#     mean_cd = np.mean(all_cd)
#     mean_cdmax = np.mean(all_cdmax)

#     print("\n========= 最终平均结果 =========")
#     print(f"平均 Chamfer Distance (CD): {mean_cd}")
#     print(f"平均 Chamfer Distance Max (CDMAX): {mean_cdmax}")
#     return mean_cd, mean_cdmax
def chamfer_distance(points_src, points_tgt):
    tree_tgt = cKDTree(points_tgt)
    distances_src, _ = tree_tgt.query(points_src)
    tree_src = cKDTree(points_src)
    distances_tgt, _ = tree_src.query(points_tgt)
    cd = (np.sum(distances_src) + np.sum(distances_tgt)) / (len(distances_src) + len(distances_tgt))
    cdmax = max(np.max(distances_src), np.max(distances_tgt))
    return cd, cdmax

def compute_hand_cd_for_subfolders(base_pred_dir, gt_dir, subfolders):
    all_cd = []
    all_cdmax = []

    # hand_indices = np.concatenate([lhand, rhand])  # 合并左右手
    # hand_indices = np.unique(hand_indices)  # 避免重复
    hand_indices =face
    for sub in subfolders:
        pred_folder = os.path.join(base_pred_dir, sub)
        pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith('.ply')])
        
        print(f"\n处理子文件夹: {sub}, 共 {len(pred_files)} 个文件")
        
        for fname in tqdm(pred_files):
            pred_path = os.path.join(pred_folder, fname)
            name_id = fname.replace('_point.ply', '').zfill(5)
            gt_name = f"{sub}_mesh-f{name_id}.ply"
            gt_path = os.path.join(gt_dir, gt_name)

            if not os.path.exists(gt_path):
                print(f"警告：找不到GT文件 {gt_path}，跳过")
                continue

            try:
                pred_points = rotate_to_xz_plane(load_ply_file(pred_path))
                gt_points = load_ply_file(gt_path)

                # 提取手部区域的点
                pred_hand = pred_points[hand_indices]
                gt_hand = gt_points[hand_indices]

                cd, cdmax = chamfer_distance(pred_hand, gt_hand)
                all_cd.append(cd)
                all_cdmax.append(cdmax)
            except Exception as e:
                print(f"跳过 {fname}，错误：{e}")
    
    mean_cd = np.mean(all_cd)
    mean_cdmax = np.mean(all_cdmax)

    print("\n========= 最终平均结果（仅手部） =========")
    print(f"平均 Hand Chamfer Distance (CD): {mean_cd}")
    print(f"平均 Hand Chamfer Distance Max (CDMAX): {mean_cdmax}")
    return mean_cd, mean_cdmax

# === 配置路径 ===
base_pred_dir = '/home/suyuze/workspace/ExAvatar_mine/0724best/result/00028/test'
gt_dir = '/home/suyuze/workspace/ExAvatar_mine/data/XHumans/data/00028/mesh'
subfolders = ['Take6', 'Take8', 'Take14']

# === 执行计算 ===
compute_hand_cd_for_subfolders(base_pred_dir, gt_dir, subfolders)
