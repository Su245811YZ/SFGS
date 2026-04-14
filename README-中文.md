[Structure-Aware Fine-Grained Gaussian Splatting for Expressive Avatar Reconstruction](https://arxiv.org/abs/2604.09324), [arXiv](https://arxiv.org/abs/2604.09324)

1. 目录结构
请确保你的文件布局如下所示，特别是 common 目录下的模型文件路径：

Plaintext
SFGS
```
${ROOT}
|-- main
|-- common
|-- |-- utils/human_model_files
|-- |-- |-- smplx/SMPLX_FEMALE.npz
|-- |-- |-- smplx/SMPLX_MALE.npz
|-- |-- |-- smplx/SMPLX_NEUTRAL.npz
|-- |-- |-- smplx/MANO_LEFT.pkl             
|-- |-- |-- smplx/MANO_RIGHT.pkl     
|-- |-- |-- smplx/MANO_SMPLX_vertex_ids.pkl
|-- |-- |-- smplx/SMPL-X__FLAME_vertex_ids.npy
|-- |-- |-- smplx/smplx_flip_correspondences.npz
|-- |-- |-- flame/flame_dynamic_embedding.npy
|-- |-- |-- flame/FLAME_FEMALE.pkl
|-- |-- |-- flame/FLAME_MALE.pkl
|-- |-- |-- flame/FLAME_NEUTRAL.pkl
|-- |-- |-- flame/flame_static_embedding.pkl
|-- |-- |-- flame/FLAME_texture.npz
|-- data
|-- |-- XHumans
|-- |-- |-- data/00028
|-- |-- |-- data/00034
|-- |-- |-- data/00087
|-- tools
|-- output
```
模型下载提示：

SMPL-X 1.1 版本

FLAME 2020 版本

2. XHumans 数据准备
https://drive.google.com/drive/folders/1TalHPkbohPoTPNawVi2gbj6M8nAyYAE9?usp=sharing


3. 训练 (Train)

进入 main 文件夹，运行以下脚本（以 00028 为例）：

Bash
CUDA_VISIBLE_DEVICES=0 python train.py --subject_id 00028
训练产生的 Checkpoints 会保存在 output/model/00028。

4. 可视化与动画制作
4.1 可视化中性姿态旋转头像

Bash
python get_neutral_pose.py --subject_id 00028 --test_epoch 20
结果可在 ./main/neutral_pose 路径下查看。

4.2 动作驱动 (Animation)
使用特定动作参数驱动头像：

Bash
python animation.py --subject_id 00028 --test_epoch 20 --motion_path $PATH
其中 $PATH 需包含驱动头像所需的 SMPL-X 参数。

若需渲染旋转相机视角效果，请运行：

Bash
python animate_view_rot.py --subject_id 00028 --test_epoch 20 --motion_path $PATH
5. 测试与评估 (Test & Eval)
运行测试渲染：

Bash
python test.py --subject_id 00028 --test_epoch 20
渲染结果将输出至 output/result/00028。

执行定量评估：
进入 tools 文件夹，运行：

Bash
python eval_xhumans.py --output_path ../output/result/00028 --subject_id 00028
