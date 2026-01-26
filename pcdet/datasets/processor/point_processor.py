import numpy as np
import torch
import math
from typing import Tuple
# import matplotlib.pyplot as plt
import sys
sys.path.append("/home/Projects/DensePillar")  # 添加上一级目录到系统路径中
# sys.path.append("/home/leon/Documents/Projects/evi-map")  # 添加上一级目录到系统路径中


from scene_reconstruction.core.volume import Volume
from scene_reconstruction.occupancy.grid import occupancy_from_points, spherical_reflection_and_transmission_from_lidar
from scene_reconstruction.math.spherical_coordinate_system import (
    cartesian_to_spherical,
    spherical_volume_element_center_and_voxel_size,
)
from scene_reconstruction.math.dempster_shafer import belief_from_reflection_and_transmission_stacked

class MapsGenerator:
    def __init__(
        self,
        cartesian_lower: Tuple[float, float, float] = (-50.0, -25.0, -2.5),
        cartesian_upper: Tuple[float, float, float] = (50.0, 25.0, 1.5),
        cartesian_shape: Tuple[int, int, int] = (1000, 500, 8),
        spherical_lower: Tuple[float, float, float] = (2.0, (90 - 5) / 180 * math.pi, -math.pi),
        spherical_upper: Tuple[float, float, float] = (60.0, (90 + 45) / 180 * math.pi, math.pi),
        spherical_shape: Tuple[int, int, int] = (600, 100, 720),
    ):
        self.cartesian_lower = cartesian_lower
        self.cartesian_upper = cartesian_upper
        self.cartesian_shape = cartesian_shape
        self.spherical_lower = spherical_lower
        self.spherical_upper = spherical_upper
        self.spherical_shape = spherical_shape

        self.point_range = np.array([-50.0, -25.0, -2.5, 50.0, 25.0, 1.5], dtype=np.float32) # 
        self.device = "cuda"
        self.pad_tensor = self.pad_vibmap().to(self.device)


    def display_parameters(self):
        """Prints all the parameters of the class."""
        print("Projection Parameters:")
        print(f"  Cartesian Lower Bound: {self.cartesian_lower}")
        print(f"  Cartesian Upper Bound: {self.cartesian_upper}")
        print(f"  Cartesian Shape: {self.cartesian_shape}")
        print(f"  Spherical Lower Bound: {self.spherical_lower}")
        print(f"  Spherical Upper Bound: {self.spherical_upper}")
        print(f"  Spherical Shape: {self.spherical_shape}")

    def pad_vibmap(self):
        # 用于填充二值图像中中心空洞
        height, width = 500, 1000
        radius = 20
        
        tensor = torch.zeros((height, width)) # 全为 0 的 Tensor
        center_y, center_x = height // 2, width // 2 # 计算中心位置坐标 (y0, x0)
        # 生成坐标网格
        y_coords = torch.arange(0, height).unsqueeze(1).expand(height, width)  # y 坐标
        x_coords = torch.arange(0, width).unsqueeze(0).expand(height, width)  # x 坐标
        
        distance_squared = (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2 # 计算到中心点的距离平方
        tensor[distance_squared <= radius ** 2] = 1 # 设置圆形区域内的值为 1

        return tensor

    def mask_points_by_range(self, points, limit_range):
        # # 根据点云的范围产生mask，过滤掉范围外的点云
        mask = (points[:, 0] >= self.point_range[0]) & (points[:, 0] <= self.point_range[3]) \
            & (points[:, 1] >= self.point_range[1]) & (points[:, 1] <= self.point_range[4])
        return mask

    def Ref_Trans_inSpherical(self, points):
        '''
        输入point, 计算参数
        '''
        lidar_min_distance: float = 1.0
        # point_path = path
        # points_lidar = np.fromfile(point_path, dtype=np.float32, count=-1).reshape(-1, 4)[:,0:3]
        points_lidar = points

        point_cloud_range = np.array([-50.0, -25.0, -2.5, 50.0, 25.0, 1.5], dtype=np.float32) # 
        mask = self.mask_points_by_range(points_lidar, point_cloud_range)
        points_lidar = points_lidar[mask]
        points_lidar = torch.from_numpy(points_lidar).unsqueeze(0) # [1, 124823, 3]
        # 过滤太近的点？
        points_mask = points_lidar.norm(dim=-1) >= lidar_min_distance # [1, 124823]
        points_weight = points_mask.unsqueeze(-1).float() # [1,124823] -> [1, 124823, 1]
        
        spherical_volume = Volume.new_volume(self.spherical_lower, self.spherical_upper)
        reflections_and_transmissions = spherical_reflection_and_transmission_from_lidar(
            points_lidar,
            points_weight = points_weight,
            spherical_volume = spherical_volume.to(points_mask.device, non_blocking=True),
            spherical_shape = self.spherical_shape,
        )  

        return reflections_and_transmissions

    def Ref_Trans_inGrid(self, points):
        '''
        2、将球形坐标转换成网格坐标[-50.0, -25.0, -2.5, 50.0, 25.0, 1.5]
        '''
        # pc_path = path
        cartesian_shape = self.cartesian_shape
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # missing_only: bool = False
        # 创建一个新体素网格
        ego_volume = Volume.new_volume(lower = self.cartesian_lower, upper = self.cartesian_upper)
        sph_volume = Volume(
            lower = torch.tensor(self.spherical_lower).unsqueeze(0),
            upper = torch.tensor(self.spherical_upper).unsqueeze(0),
        )
        scene_instance_index = torch.zeros(1, *cartesian_shape, dtype=torch.long) # [1, 1000, 500, 8]
        sample_points = ego_volume.coord_grid(scene_instance_index) # [1, 1000, 500, 32, 3] # 3:voxl中心点的坐标，like[ 49.9500, -24.7500,   1.2500]
        # frame_lidar_points_valid = torch.where(valid_transform[..., None], frame_lidar_points, float("nan")) # 并非笛卡尔网格中的每个点都可以映射到球面网格

        sample_points_spherical = cartesian_to_spherical(sample_points) # [1, 500, 1000, 32, xyz] -> [1, 500, 1000, 32, r theta phi] 
        # spherical_rt = ReflectionTransmissionSpherical(path = pc_path, ref_parms = Parms).to(device) # [1, 2, 600, 100, 720] [B, C, X, Y, Z]
        spherical_rt = self.Ref_Trans_inSpherical(points)
        cartesian_rt = sph_volume.sample_volume(spherical_rt, sample_points_spherical) # [1, 2, 400, 400, 32]
        
        spherical_voxel_size = sph_volume.voxel_size(spherical_rt) # [1, 3]
        cart_voxel_size = ego_volume.voxel_size(cartesian_rt) # [1, 3]
        scale = (
            cart_voxel_size.prod(-1)[:, None, None, None]
            / spherical_volume_element_center_and_voxel_size(
                sample_points_spherical, spherical_voxel_size[:, None, None, None, :]
            )
        ).nan_to_num(0.0)
        cartesian_rt_scaled = scale.unsqueeze(1) * cartesian_rt # [1, 2, 400, 400, 32]
        assert cartesian_rt_scaled.shape[0] == 1, "Only one ref frame allowed"

        return cartesian_rt_scaled.transpose(2, 3)

    def gene_map(self, refl_and_trans):
        # 计算置信度
        bba_occupied_free = belief_from_reflection_and_transmission_stacked(
            refl_and_trans, p_fn=0.95, p_fp=0.05 # probabilites from config/bba.yaml for a voxel size of 0.2
        )
        bba_omega = 1.0 - bba_occupied_free.sum(dim=1, keepdim=True)
        bba_omega_mean = 1.0 - bba_omega[0, 0, :, :, :].mean(dim=-1)
        binary_map = (bba_omega_mean > 0.0005).int()
        visb_tensor = torch.clamp(self.pad_tensor + binary_map, 0, 1) 

        occu_map = bba_omega_mean.unsqueeze(-1)
        visb_map = visb_tensor.unsqueeze(-1)

        return occu_map, visb_map