# PointDenseBEV: Dense Semantic Bird’s-Eye-View Map Generation

> This is the code implementation of IROS2025 article **Dense Semantic Bird’s-Eye-View Map Generation from Sparse LiDAR Point Clouds via Distribution-Aware Feature Fusion**

[![PointDenseBEV Demonstration](https://img.youtube.com/vi/hv8dzhZIFgk/0.jpg)](https://youtu.be/hv8dzhZIFgk)

*Click the image above to watch the demonstration video.*

## 📖 Overview
We propose **PointDenseBEV**, an end-to-end, distribution-aware feature fusion framework. It takes sparse LiDAR point clouds as input and directly generates dense semantic Bird's-Eye-View (BEV) maps. Spatial geometric information and temporal context are embedded as auxiliary semantic cues within the BEV grid representation to improve semantic density. 

Extensive experiments on the **SemanticKITTI** dataset demonstrate that our method achieves competitive performance compared with existing approaches.

> **Note:** This project will continue to be refined in the near future, including environment mirroring, data preprocessing, and weighting results. 

### 🔗 Resources
- **Base Framework:** Built on top of [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).
- **Project Files & Dataset:** [Download via OneDrive](https://1drv.ms/f/c/e066035012049ad0/IgAZUSPHZCS1S4ZxWTp8f6CYAVgT0lJ-kSyEAMLHotsKsAE?e=zLYiQf)

---

## 🚀 Getting Started

### 1. Environment Setup (Docker)
We strongly recommend using Docker to set up the environment. You can download the Docker environment package (`segkitti_250114.tar`) from the provided OneDrive link and build the image following the standard Docker guide.

Run the following command to start the container:

```bash
docker run --name seg_kitti \
    --shm-size=128g \
    --gpus all \
    -dit \
    -p 9800:9800 \
    -v /path/to/your/Datasets:/home/Datasets \
    -v /path/to/your/Projects:/home/Projects \
    c3a505511a3f \
    /bin/bash
```
*(Replace `/path/to/your/...` and the image ID `c3a505511a3f` with your local paths and actual loaded image ID if necessary).*

### 2. Installation
Since this project is built upon `OpenPCDet`, it needs to be installed before running.

```bash
# Clone the repository (if not already done) and enter the project root
cd <PROJECT_ROOT>

# Install the project
python setup.py develop
```

---

## 📂 Data Preparation

### Download
1. Download the density maps (`kitti_vis_8`) and ground-truth labels (`dense_label`, `sparse_label`) from the provided OneDrive link.
2. Obtain the raw point clouds from the official [KITTI website](http://www.cvlibs.net/datasets/kitti/).

### Directory Structure
After decompression, organize your dataset directory as follows:

```text
<PROJECT_ROOT>
└── datasets/
    └── dense_kitti/
        ├── dense_label/        # Dense-mode ground truth
        │   ├── 00/
        │   │   └── *.bin
        │   ├── ...
        │   └── 10/
        ├── kitti_odo/          # Raw point clouds
        │   ├── 00/
        │   ├── ...
        │   └── 10/
        ├── kitti_vis_8/        # Visibility map files
        │   └── ...
        └── sparse_label/
            └── ...
```

---

## 🏃 Training and Evaluation

### Training
Run the training script to train the model. 

```bash
sh tools/train.sh
```

**Configuration Options:**
You can modify the model and dataset configurations in the following files:
- `tools/cfgs/kitti_models/densepillar.yaml`
- `tools/cfgs/dataset_configs/test_pillarnet_128.yaml`

*Example configuration modifications:*

**1. Update dataset paths** (`kitti_dataset.yaml`):
```yaml
DATA_LOAD_PATH: '/home/Projects/DensePillar/datasets/dense_kitti'
DATA_TRAIN_PATH: '/home/Projects/DensePillar/datasets/dense_kitti/train_sample.pkl'
DATA_TEST_PATH: '/home/Projects/DensePillar/datasets/dense_kitti/test_sample.pkl'
```

**2. Enable multi-frame point cloud data**:
```yaml
POINT_MERGE_HISFRAME: 0
```

**3. Adjust voxel numbers** (`test_pillarnet_128.yaml`):
```yaml
MAX_POINTS_PER_VOXEL: 10
MAX_NUMBER_OF_VOXELS: {
    'train': 50000, 
    'test': 50000
}
```

### Evaluation
We provide pre-trained weights for single-frame LiDAR point cloud input. 
1. Download `checkpoint_epoch_11.pth` from the OneDrive link.
2. Place it in the following directory: `logs/model/kitti_models/test_pillarnet_128/pretrain/ckpt/`
3. Evaluate the trained model (make sure to set the `ckpt_dir` path properly inside the script):

```bash
sh tools/test.sh
```

> **Visualization:** If you want to visualize the evaluation results, uncomment the visualization code block in `tools/eval_utils/eval_utils.py`.

---

## 📚 Citation

If you find our work helpful, please cite:

```
@INPROCEEDINGS{li2025dense,
  author={Jinsong Li and Kunyu Peng and Yuxiang Sun},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Dense Semantic Bird-Eye-View Map Generation from Sparse LiDAR Point Clouds via Distribution-aware Feature Fusion}, 
  year={2025},
  volume={},
  number={},
  pages={4123-4129},
  doi={10.1109/IROS60139.2025.11246382}}
```
---

## 🙏 Acknowledgements
Special thanks to:
- **[OpenPCDet](https://github.com/open-mmlab/OpenPCDet)** for the highly extensible 3D object detection codebase.
- **MASS (Multi-Attentional Semantic Segmentation of LiDAR Data for Dense Top-View Understanding)** for their inspiring contributions to the community.
