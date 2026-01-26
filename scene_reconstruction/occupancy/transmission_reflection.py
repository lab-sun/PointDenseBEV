"""Transmission and reflections from lidar."""

from __future__ import annotations
from IPython.display import display
import math
import pprint
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import tqdm
from torch import Tensor

from scene_reconstruction.core.volume import Volume
from scene_reconstruction.data.nuscenes.dataset import NuscenesDataset
from scene_reconstruction.data.nuscenes.polars_helpers import series_to_torch, torch_to_series
from scene_reconstruction.occupancy.grid import occupancy_from_points, spherical_reflection_and_transmission_from_lidar


def _batch(iterable, n=1):
    lenght = len(iterable)
    for ndx in range(0, lenght, n):
        yield iterable[ndx : min(ndx + n, lenght)]

import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
import torch
print(torch.__version__)
# pytorch = "*"
# cuda = ">=11.3,<12"
# torchvision = "0.13.1"
@dataclass
class ReflectionTransmissionSpherical:
    """Calculates number of transmission and reflectiones per voxel cell from lidar pointcloud in spherical coordinates in the lidar frame."""
    '''在激光雷达框架内以球面坐标计算激光雷达点云每个体素单元的透射和反射次数'''
    ds: NuscenesDataset
    extra_data_root: Path
    spherical_lower: tuple[float, float, float] = (2.0, (90 - 15) / 180 * math.pi, -math.pi)
    spherical_upper: tuple[float, float, float] = (60.0, (90 + 35) / 180 * math.pi, math.pi)
    spherical_shape: tuple[int, int, int] = (600, 100, 720)
    lidar_min_distance: float = 2.0
    # voxel_size sperical : 0.1m, 0.5°, 0.5°

    batch_size: int = 1
    device: str = "cuda"
    name: str = "reflection_and_transmission_spherical"

    def process_data(self) -> None:
        """Process dataset."""
        for scene in tqdm.tqdm(
            self.ds.scene.iter_slices(1), total=len(self.ds.scene), position=0, desc="Processing scenes"
        ):
            self.process_scene(scene)

    def save_path(self, scene_name: str, lidar_top_token: str) -> Path:
        """Save from scene name and lidar token."""
        path = Path(self.extra_data_root) / self.name / scene_name / "LIDAR_TOP" / f"{lidar_top_token}.arrow"
        path.parent.mkdir(exist_ok=True, parents=True)
        return path

    def process_scene(self, scene: pl.DataFrame) -> None:
        """Process single scene."""
        # load all sample data with lidar token
        scene = self.ds.join(scene, self.ds.sample)
        scene = self.ds.load_sample_data(scene, "LIDAR_TOP", with_data=False)
        spherical_volume = Volume.new_volume(self.spherical_lower, self.spherical_upper)
        
        volume_lower = torch_to_series(
            "LIDAR_TOP.reflection_and_transmission_spherical.volume.lower", spherical_volume.lower
        )
        volume_upper = torch_to_series(
            "LIDAR_TOP.reflection_and_transmission_spherical.volume.upper", spherical_volume.upper
        )
        # nuscenes点云数据有五个通道分别表示x,y,z强度（0~255的整数值）和激光束编号（0~31的整数）
        # kitt x, y, z, r（强度，反射
        for chunked_scene in tqdm.tqdm(
            scene.iter_slices(self.batch_size),
            total=(len(scene) + self.batch_size - 1) // self.batch_size, # 4
            desc="Processing sample",
            position=1,
        ):
            '''chunked_scene 为class 'polars.dataframe.frame.DataFrame'''
            chunked_scene = self.ds._load_sensor_data(chunked_scene, "LIDAR_TOP") # (4, 39) 39个类别索引   
            points_lidar = series_to_torch(chunked_scene["LIDAR_TOP.sample_data.points_lidar"]).to(
                self.device, non_blocking=True
            ) # torch.Size([4, 35000, 3]) 

            # ##
            # pc_path='/home/leon/Documents/Projects/evi-map/data/kitti/01/000000test.bin' # 000101
            # points_lidar = np.fromfile(pc_path, dtype=np.float32, count=-1).reshape(-1, 5)[:,0:3]
            # points_lidar = torch.from_numpy(points_lidar).to("cuda", non_blocking=True).unsqueeze(0) # [1, 124823, 3]
            # ##

            points_range_mask = points_lidar.norm(dim=-1) >= self.lidar_min_distance
            # print("ddd",points_range_mask[0].sum())
            # result = points_range_mask[0] & series_to_torch(chunked_scene["LIDAR_TOP.sample_data.points_mask"])[0].to(self.device, non_blocking=True)
            # print("result.sum()",result.sum())

            points_mask = (
                series_to_torch(chunked_scene["LIDAR_TOP.sample_data.points_mask"]).to(self.device, non_blocking=True)
                & points_range_mask
            ) # ([4, 35000])
            # points_mask = points_lidar.norm(dim=-1) >= self.lidar_min_distance # [1, 124823]

            '''计算反射和透射'''
            reflections_and_transmissions = self.process_batch(
                points_lidar, points_mask, spherical_volume.to(points_mask.device, non_blocking=True)
            )
            # print("reflections_and_transmissions",reflections_and_transmissions[0].sum()) # torch.Size([4, 2, 600, 100, 720])

            for sample, rt in zip(
                chunked_scene.select(
                    "scene.token", "sample.token", "scene.name", "LIDAR_TOP.sample_data.token"
                ).iter_slices(1),
                reflections_and_transmissions.split(1),
            ):
                '''取连续四帧，计算穿透和反射，再把结果给每一帧，并没有固定用前面几帧或者后面几帧'''
                filename = self.save_path(sample["scene.name"].item(), sample["LIDAR_TOP.sample_data.token"].item())
                '''print("np.shape(rt)",np.shape(rt)) # torch.Size([1, 2, 600, 100, 720])'''
                sample = sample.select("LIDAR_TOP.sample_data.token").with_columns(
                    torch_to_series(f"LIDAR_TOP.{self.name}", rt),
                    volume_lower,
                    volume_upper,
                )
                '''保存为.arrow'''
                sample.write_ipc(filename, compression="zstd")

    def process_batch(self, points_lidar: Tensor, points_mask: Tensor, spherical_volume: Volume) -> Tensor:
        """Process single batch of data."""
        # points_lidar: B, N, 3
        # points_mask: B, N
        # lidar_from_ego: B, 4, 4

        points_weight = points_mask.unsqueeze(-1).float() # [4, 35000] -> [4, 35000, 1]
        reflections_and_transmissions = spherical_reflection_and_transmission_from_lidar(
            points_lidar,
            points_weight=points_weight,
            spherical_volume=spherical_volume,
            spherical_shape=self.spherical_shape,
        )
        return reflections_and_transmissions.to(device="cpu", non_blocking=True)  # .cpu()  # [B, 2, X, Y, Z]

# @dataclass
# class TransmissionAndReflections:
#     """Calculates number of transmission and reflectiones per voxel cell from lidar pointcloud."""
#     '''从激光雷达点云计算每个体素单元的透射和反射次数'''
#     ds: NuscenesDataset
#     extra_data_root: Path
#     cartesian_lower: tuple[float, float, float] = (-40.0, -40.0, -1.0)
#     cartesian_upper: tuple[float, float, float] = (40.0, 40.0, 5.4)
#     cartesian_shape: tuple[int, int, int] = (400, 400, 32)
#     spherical_lower: tuple[float, float, float] = (0.0, (90 - 15) / 180 * math.pi, -math.pi)
#     spherical_upper: tuple[float, float, float] = (60.0, (90 + 35) / 180 * math.pi, math.pi)
#     spherical_shape: tuple[int, int, int] = (600, 100, 720)
#     lidar_min_distance: float = 3.0
#     # voxel_size cartesian = 0.2m
#     # voxel_size sperical : 0.1m, 0.5°, 0.5°

#     batch_size: int = 4

#     def process_data(self) -> None:
#         """Process dataset."""
#         for scene in tqdm.tqdm(self.ds.scene.partition_by("scene.token"), position=0):
#             scene = self.ds.join(scene, self.ds.sample)
#             scene = self.ds.load_sample_data(scene, "LIDAR_TOP")
#             self.process_scene(scene)

#     def save_path(self, scene_name: str, lidar_top_token: str) -> Path:
#         """Save from scene name and lidar token."""
#         path = (
#             Path(self.extra_data_root)
#             / scene_name
#             / "reflection_and_transmission"
#             / "LIDAR_TOP"
#             / f"{lidar_top_token}.npz"
#         )
#         path.parent.mkdir(exist_ok=True, parents=True)
#         return path

#     def save_data(self, save_filename: Path, reflections_and_transmissions: Tensor):
#         """Save data."""
#         num_reflections, num_transmissions = reflections_and_transmissions.unbind(0)
#         np.savez_compressed(
#             save_filename,
#             num_reflections=num_reflections.numpy(),
#             num_transmissions=num_transmissions.numpy(),
#         )

#     def process_scene(self, scene: pl.DataFrame) -> None:
#         """Process single scene."""
#         points_lidar = series_to_torch(scene["LIDAR_TOP.sample_data.points_lidar"])
#         ego_from_lidar = series_to_torch(scene["LIDAR_TOP.transform.ego_from_sensor"])
#         lidar_from_ego = ego_from_lidar.inverse()
#         points_range_mask = points_lidar.norm(dim=-1) >= self.lidar_min_distance
#         points_mask = series_to_torch(scene["LIDAR_TOP.sample_data.points_mask"]) & points_range_mask
#         filename = [
#             self.save_path(s_name, l_token)
#             for s_name, l_token in zip(scene["scene.name"], scene["LIDAR_TOP.sample_data.token"])
#         ]
#         for points_lidar_batched, lidar_from_ego_batched, points_mask_batched, filename_batched in tqdm.tqdm(
#             zip(
#                 points_lidar.split(self.batch_size),
#                 lidar_from_ego.split(self.batch_size),
#                 points_mask.split(self.batch_size),
#                 _batch(filename, self.batch_size),
#             ),
#             total=(points_lidar.shape[0] + self.batch_size - 1) // self.batch_size,
#             position=1,
#         ):
#             reflections_and_transmissions_batched = self.process_batch(
#                 points_lidar_batched.cuda(),
#                 lidar_from_ego_batched.cuda(),
#                 points_mask_batched.cuda(),
#             )

#             for filename, reflections_and_transmissions in zip(
#                 filename_batched, reflections_and_transmissions_batched.cpu().unbind(0)
#             ):
#                 self.save_data(filename, reflections_and_transmissions)  # type: ignore

#     def process_batch(self, points_lidar: Tensor, lidar_from_ego: Tensor, points_mask: Tensor) -> Tensor:
#         """Process single batch of data."""
#         # points_lidar: B, N, 3
#         # points_mask: B, N
#         # lidar_from_ego: B, 4, 4
#         points_weight = points_mask.unsqueeze(-1).float()
#         reflections_and_transmissions = occupancy_from_points(
#             points_lidar,
#             lidar_from_ego=lidar_from_ego,
#             points_weight=points_weight,
#             cartesian_volume=Volume.new_volume(self.cartesian_lower, self.cartesian_upper, device=points_lidar.device),
#             spherical_volume=Volume.new_volume(self.spherical_lower, self.spherical_upper, device=points_lidar.device),
#             cartesian_shape=self.cartesian_shape,
#             spherical_shape=self.spherical_shape,
#         )
#         return reflections_and_transmissions  # [B, 2, X, Y, Z]


