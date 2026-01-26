# Dense Semantic Bird’s-Eye-View Map Generation from Sparse LiDAR Point Clouds via Distribution-Aware Feature Fusion

The project will continue to be refined in the near future, including environment mirroring, data preprocessing, and weighting results. This demonstration video:

[![PointDenseBEV](https://img.youtube.com/vi/hv8dzhZIFgk/0.jpg)](https://youtu.be/hv8dzhZIFgk)

We propose **PointDenseBEV**, an end-to-end, distribution-aware feature fusion framework. It takes sparse LiDAR point clouds as input and directly generates dense semantic BEV maps. Spatial geometric information and temporal context are embedded as auxiliary semantic cues within the BEV grid representation to improve semantic density. Extensive experiments on the **SemanticKITTI** dataset demonstrate that our method achieves competitive performance compared with existing approaches.

This project is built on top of **OpenPCDet**: https://github.com/open-mmlab/OpenPCDet.

# Environment
## Docker image
You can download the Docker environment package and build the image following the Docker guide.

## Installation
Install this project with:
```
cd <PROJECT_ROOT>
python setup.py develop
```


# Data preparation
## Download
You can download the density maps (`kitti_vis_8`) and ground-truth labels (`dense_label`, `sparse_label`) from the provided links. The raw point clouds can be obtained from the official KITTI website.

## Directory structure
After decompression, the dataset directory should look like:

```
Project root
  |-- datasets
      |-- dense_kitti
          |-- dense_label        <Dense-mode ground truth>
          |   |-- 00
          |   |   |-- *.bin
          |   |-- ...
          |   |-- 10
          |-- kitti_odo          <Raw point clouds>
          |   |-- 00
          |   |-- ...
          |   |-- 10
          |-- kitti_vis_8        <Visibility map files>
          |   |-- ...
          |-- sparse_label
          |   |-- ...
```

# Training and evaluation
## Training
Run `tools/train.sh` to train the model. You can modify model and dataset configurations in:
- `tools/cfgs/kitti_models/densepillar.yaml`
- `tools/cfgs/dataset_configs/kitti_dataset.yaml`

```
sh tools/train.sh
```

## Evaluation
Evaluate a trained model with `tools/test.sh` (make sure to set the `ckpt_dir` path first).
```
sh tools/test.sh
```

## Acknowledgements
Thanks to **OpenPCDet** and **MASS: Multi-Attentional Semantic Segmentation of LiDAR Data for Dense Top-View Understanding** for their contributions to the community.

