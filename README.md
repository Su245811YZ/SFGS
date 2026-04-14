## [Structure-Aware Fine-Grained Gaussian Splatting for Expressive Avatar Reconstruction](https://arxiv.org/abs/2604.09324), [arXiv](https://arxiv.org/abs/2604.09324)

---

## 1. Directory Structure

Please organize your project directory as follows. In particular, make sure the model files are correctly placed under the `common` directory:

```
SFGS
${ROOT}
|-- main
|-- common
|   |-- utils/human_model_files
|       |-- smplx/SMPLX_FEMALE.npz
|       |-- smplx/SMPLX_MALE.npz
|       |-- smplx/SMPLX_NEUTRAL.npz
|       |-- smplx/MANO_LEFT.pkl             
|       |-- smplx/MANO_RIGHT.pkl     
|       |-- smplx/MANO_SMPLX_vertex_ids.pkl
|       |-- smplx/SMPL-X__FLAME_vertex_ids.npy
|       |-- smplx/smplx_flip_correspondences.npz
|       |-- flame/flame_dynamic_embedding.npy
|       |-- flame/FLAME_FEMALE.pkl
|       |-- flame/FLAME_MALE.pkl
|       |-- flame/FLAME_NEUTRAL.pkl
|       |-- flame/flame_static_embedding.pkl
|       |-- flame/FLAME_texture.npz
|-- data
|   |-- XHumans
|       |-- data/00028
|       |-- data/00034
|       |-- data/00087
|-- tools
|-- output
```

### Model Downloads

* **SMPL-X**: Version 1.1
* **FLAME**: Version 2020

---

## 2. XHumans Data Preparation

Download the dataset from the following link:

* [https://drive.google.com/drive/folders/1TalHPkbohPoTPNawVi2gbj6M8nAyYAE9?usp=sharing](https://drive.google.com/drive/folders/1TalHPkbohPoTPNawVi2gbj6M8nAyYAE9?usp=sharing)

---

## 3. Training

Navigate to the `main` directory and run the following command (taking subject `00028` as an example):

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --subject_id 00028
```

The trained checkpoints will be saved to:

```
output/model/00028
```

---

## 4. Visualization and Animation

### 4.1 Neutral Pose Visualization

To render a rotating avatar in the neutral pose:

```bash
python get_neutral_pose.py --subject_id 00028 --test_epoch 20
```

The results will be saved under:

```
./main/neutral_pose
```

---

### 4.2 Motion-Driven Animation

To animate the avatar using motion parameters:

```bash
python animation.py --subject_id 00028 --test_epoch 20 --motion_path $PATH
```

* `$PATH` should contain the SMPL-X parameters used to drive the avatar.

To render animation with a rotating camera view:

```bash
python animate_view_rot.py --subject_id 00028 --test_epoch 20 --motion_path $PATH
```

---

## 5. Testing and Evaluation

### Rendering Results

```bash
python test.py --subject_id 00028 --test_epoch 20
```

The rendered results will be saved to:

```
output/result/00028
```

---

### Quantitative Evaluation

Navigate to the `tools` directory and run:

```bash
python eval_xhumans.py --output_path ../output/result/00028 --subject_id 00028
```

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{su2026structureawarefinegrainedgaussiansplatting,
  title={Structure-Aware Fine-Grained Gaussian Splatting for Expressive Avatar Reconstruction}, 
  author={Yuze Su and Hongsong Wang and Jie Gui and Liang Wang},
  year={2026},
  eprint={2604.09324},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2604.09324}, 
}
```


