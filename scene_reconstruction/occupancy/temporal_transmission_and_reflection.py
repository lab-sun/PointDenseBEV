"""Temporal accumulation of transmission and reflection."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np

import polars as pl
import torch
import tqdm
from torch import Tensor

from scene_reconstruction.core import einsum_transform
from scene_reconstruction.core.volume import Volume
from scene_reconstruction.data.nuscenes.dataset import NuscenesDataset
from scene_reconstruction.data.nuscenes.polars_helpers import series_to_torch, torch_to_series
from scene_reconstruction.data.nuscenes.scene_utils import scene_to_tensor_dict
from scene_reconstruction.math.spherical_coordinate_system import (
    cartesian_to_spherical,
    spherical_volume_element_center_and_voxel_size,
)


def _load_flow_and_rt(ds: NuscenesDataset, scene: pl.DataFrame, extra_data_root: Union[Path, str]):
    scene = ds.load_reflection_and_transmission_spherical(scene, extra_data_root)
    scene = ds.load_scene_flow_polars(scene, extra_data_root)
    return scene


class AccumulateFrames:
    """Accumlate reflection and transmission for reference frame."""

    def __init__(
        self,
        ds: NuscenesDataset,
        ref_frame: pl.DataFrame,
        extra_data_root: Union[Path, str],
        icp_alignment: bool = False,
        batch_size: int = 2,
        max_num_frames: int = 50,
        max_ego_pose_difference: float = 20.0,
        device: str = "cuda",
        num_threads: int = 8,
    ) -> None:
        """Initialize refernce frame."""
        self.extra_data_root = extra_data_root
        self.kwargs = {"device": device, "non_blocking": True}
        self.ds = ds
        self.use_icp_alignment = icp_alignment
        self.frame_dict = {k: v.to(**self.kwargs) for k, v in self._load_fn(ref_frame).items()}
        # print("self.frame_dict ",np.shape(self.frame_dict['frame_instance_ids']) )
        self.instance_ids = self.frame_dict["frame_instance_ids"]
        self.instance_from_global = self.frame_dict["frame_instance_from_global"]
        self.global_from_ego = series_to_torch(ref_frame["LIDAR_TOP.transform.global_from_ego"]).to(**self.kwargs)
        self.instance_from_ego = torch.einsum("bnig,bge->bnie", self.instance_from_global, self.global_from_ego)
        self.sph_volume = Volume(
            lower=series_to_torch(ref_frame["LIDAR_TOP.reflection_and_transmission_spherical.volume.lower"]).to(
                **self.kwargs
            ),
            upper=series_to_torch(ref_frame["LIDAR_TOP.reflection_and_transmission_spherical.volume.upper"]).to(
                **self.kwargs
            ),
        )
        # print("ref_frameLIDAR_TOP.reflection_and_transmission_spherical.volume.lower",ref_frame["LIDAR_TOP.reflection_and_transmission_spherical.volume.lower"])
        # print("ref_frameLIDAR_TOP.reflection_and_transmission_spherical.volume.upper",ref_frame["LIDAR_TOP.reflection_and_transmission_spherical.volume.upper"])
        self.ego_volume = Volume(
            lower=series_to_torch(ref_frame["LIDAR_TOP.scene_flow.volume.lower"]).to(**self.kwargs),
            upper=series_to_torch(ref_frame["LIDAR_TOP.scene_flow.volume.upper"]).to(**self.kwargs),
        )
        # print("ddd",ref_frame["LIDAR_TOP.scene_flow.volume.lower"])
        self.max_ego_pose_difference = max_ego_pose_difference
        self.max_num_frames = max_num_frames
        self.batch_size = batch_size
        self.num_threads = num_threads

        self._init_ref_frame()

    def _init_ref_frame(self):
        sample_points = self._generate_sample_points(
            self.frame_dict["frame_instance_ids"], # [1, 400, 400, 32]
            self.frame_dict["frame_instance_from_global"], # [1, 228, 4, 4]
            self.frame_dict["frame_ego_from_global"], # [1, 4, 4]
            self.frame_dict["frame_lidar_from_ego"], # frame_lidar_from_ego[1, 4, 4]
        ) # [1, 400, 400, 32, 3] 
        sample_points_spherical = cartesian_to_spherical(sample_points)
        spherical_rt = self.frame_dict["spherical_rt"] # spherical_rt[1, 2, 600, 100, 720]
        # print("init")
        cartesian_rt = self.sph_volume.sample_volume(spherical_rt, sample_points_spherical) # [1, 2, 400, 400, 32]
        # print("end")
        spherical_voxel_size = self.sph_volume.voxel_size(spherical_rt) # [1, 3]
        cart_voxel_size = self.ego_volume.voxel_size(cartesian_rt) # [1, 3]
        # there might be points containing "nan" values, since they have no valid transformation
        # 可能存在包含 “nan” 值的点，因为它们没有有效的变换
        scale = (
            cart_voxel_size.prod(-1)[:, None, None, None]
            / spherical_volume_element_center_and_voxel_size(
                sample_points_spherical, spherical_voxel_size[:, None, None, None, :]
            )
        ).nan_to_num(0.0)
        cartesian_rt_scaled = scale.unsqueeze(1) * cartesian_rt # [1, 2, 400, 400, 32]
        assert cartesian_rt_scaled.shape[0] == 1, "Only one ref frame allowed"
        self.agg_rt = cartesian_rt_scaled.sum(0, keepdim=True) # [1, 2, 400, 400, 32]
        self.num_samples = cartesian_rt_scaled.shape[0] # 1


    def _generate_sample_points(
        self,
        frame_instance_ids: Tensor, # [1, 400, 400, 32] frame_instance_ids = scene_instance_index,
        frame_instance_from_global: Tensor,
        frame_ego_from_global: Tensor,
        frame_lidar_from_ego: Tensor,
    ):
        # 变换矩阵
        frame_global_from_instance = frame_instance_from_global.inverse() # [1, 228, 4, 4]
        # print("frame_global_from_instance",np.shape(frame_global_from_instance))
        if self.use_icp_alignment:
            raise NotImplementedError()

        # 合成对应坐标变换矩阵
        frame_lidar_from_global = frame_lidar_from_ego @ frame_ego_from_global # [1, 4, 4]
        frame_ego_from_instance = torch.einsum("beg,bngi->bnei", frame_ego_from_global, frame_global_from_instance) # [1, 228, 4, 4]
        frame_lidar_from_instance = torch.einsum("blg,bngi->bnli", frame_lidar_from_global, frame_global_from_instance) # [1, 228, 4, 4]
        # 处理自车坐标系中的实例映射
        frame_ego_from_self_ego = frame_ego_from_instance @ self.instance_from_ego # [1, 228, 4, 4]
        frame_lidar_from_self_ego = frame_lidar_from_instance @ self.instance_from_ego # [1, 228, 4, 4]
        self_instance_ids_flat = self.instance_ids.flatten(1) # [1, 400, 400, 32]->[1, 5120000]

        # 
        cartesian_shape: tuple[int, int, int] = (400, 400, 32)
        frame_instance_ids = torch.zeros(1, *cartesian_shape, dtype=torch.long).to("cuda")

        # 对变换矩阵进行实例 ID 的索引操作
        frame_ego_from_self_ego_gathered_flat = frame_ego_from_self_ego.gather(
            1, self_instance_ids_flat[..., None, None].expand(frame_instance_ids.shape[0], -1, 4, 4)
        ) # [1, 5120000, 4, 4] None填充[4,4]
        frame_ego_from_self_ego_dense = frame_ego_from_self_ego_gathered_flat.view(
            -1, *self.instance_ids.shape[1:], 4, 4
        ) # [1, 400, 400, 32, 4, 4]
        # 生成自车坐标系中的点云
        # testpoints=self.ego_volume.coord_grid(frame_instance_ids)
        # print("testpoints",testpoints[0,0,0])

        frame_ego_points = einsum_transform(
            "bxyzfs,bxyzs->bxyzf", 
            frame_ego_from_self_ego_dense, 
            points=self.ego_volume.coord_grid(frame_instance_ids)
        ) # [1, 400, 400, 32, 3]
        # sample ids to check for valid transforms
        self_instance_ids_sampled = self.ego_volume.sample_volume(
            frame_instance_ids.unsqueeze(1).float(), frame_ego_points, mode="nearest", fill_invalid=float("nan")
        ) # [1, 1, 400, 400, 32]

        valid_transform = self_instance_ids_sampled.squeeze(1) == self.instance_ids.float()  # [B, X, Y, Z] [1, 400, 400, 32]

        frame_lidar_from_self_ego_gathered_flat = frame_lidar_from_self_ego.gather(
            1, self_instance_ids_flat[..., None, None].expand(frame_instance_ids.shape[0], -1, 4, 4)
        ) # [1, 5120000, 4, 4]
        frame_lidar_from_self_ego_dense = frame_lidar_from_self_ego_gathered_flat.view(
            -1, *self.instance_ids.shape[1:], 4, 4
        ) # [1, 400, 400, 32, 4, 4]
        frame_lidar_points = einsum_transform(
            "bxyzfs,bxyzs->bxyzf",
            frame_lidar_from_self_ego_dense,
            points=self.ego_volume.coord_grid(frame_instance_ids),
        ) # [1, 400, 400, 32, 3]

        frame_lidar_points_valid = torch.where(valid_transform[..., None], frame_lidar_points, float("nan")) # [1, 400, 400, 32, 3]
        # print("frame_lidar_points_valid",frame_lidar_points_valid[0,0,0])
        return frame_lidar_points_valid

    def _load_fn(self, frame: pl.DataFrame):
        frame = _load_flow_and_rt(self.ds, frame, self.extra_data_root)
        frame_dict = scene_to_tensor_dict(
            frame,
            {
                "frame_instance_ids": "LIDAR_TOP.scene_flow.scene_instance_index",
                "frame_instance_from_global": "LIDAR_TOP.scene_flow.instance_from_global",
                "frame_ego_from_global": "LIDAR_TOP.transform.ego_from_global",
                "frame_lidar_from_ego": "LIDAR_TOP.transform.sensor_from_ego",
                "spherical_rt": "LIDAR_TOP.reflection_and_transmission_spherical", # 折射反射信息
            },
        )
        return frame_dict

    def _accumulate_frames(
        self,
        frame_instance_ids: Tensor,  # [1, 400, 400, 32]
        frame_instance_from_global: Tensor, # [1, 228, 4, 4]
        frame_ego_from_global: Tensor, # [1, 4, 4]
        frame_lidar_from_ego: Tensor, # [1, 4, 4]
        spherical_rt: Tensor, # [1, 2, 600, 100, 720]
    ):
        # 堆叠历史帧信息
        sample_points = self._generate_sample_points(
            frame_instance_ids,
            frame_instance_from_global,
            frame_ego_from_global,
            frame_lidar_from_ego,
        ) # torch.Size([1, 400, 400, 32, 3])  
        # 将直角坐标转换为球面坐标。

        # ####test------------------------------------------------------------------------------
        # cartesian_lower: tuple[float, float, float] = (-40.0, -40.0, -1.0)
        # cartesian_upper: tuple[float, float, float] = (40.0, 40.0, 5.4)
        # cartesian_shape: tuple[int, int, int] = (400, 400, 32)
        # ego_volume = Volume.new_volume(lower=cartesian_lower, upper=cartesian_upper)
        # scene_instance_index = torch.zeros(1, *cartesian_shape, dtype=torch.long) # [1, 1000, 500, 8]
        # frame_ego_points = ego_volume.coord_grid(scene_instance_index) # [1, 1000, 500, 32, 3] # 3:voxl中心点的坐标，like[ 49.9500, -24.7500,   1.2500]
        # sample_points = frame_ego_points
        # sample_points_spherical = cartesian_to_spherical(sample_points).to("cuda")  # torch.Size([1, 400, 400, 32, 3])
        # import sys
        # sys.path.append('/home/leon/Documents/Projects/evi-map/scene_reconstruction/core/')
        # from transform import transform_to_grid_sample_coords,to_homogenous,from_homogenous
        # normalized_from_grid = transform_to_grid_sample_coords(torch.tensor(cartesian_lower), torch.tensor(cartesian_upper) )
        # normalized_from_grid = normalized_from_grid.unsqueeze(0).to('cuda')
        # # print("np.shape(normalized_from_grid)",np.shape(normalized_from_grid),normalized_from_grid)
        # points = to_homogenous(sample_points_spherical).to('cuda')
        # points = torch.einsum("bng,bxyzg->bxyzn", normalized_from_grid, points)
        # normalized_coords = from_homogenous(points)
        # # normalized_coords = einsum_transform("bng,bxyzg->bxyzn", normalized_from_grid, points=sample_points_spherical) # [1, 1000, 500, 8, 3]
        # print("normalized_coords ddd",np.shape(normalized_coords),normalized_coords[0,35,159]) # [1, 400, 400, 8, 3]
        # ####test------------------------------------------------------------------------------

        sample_points_spherical = cartesian_to_spherical(sample_points) # torch.Size([1, 400, 400, 32, 3])
        
        # pc_path='/home/leon/Documents/Projects/evi-map/data/kitti/01/000000test.bin' # 000101
        # points_lidar = np.fromfile(pc_path, dtype=np.float32, count=-1).reshape(-1, 5)[:,0:3]
        # points_lidar = torch.from_numpy(points_lidar).to("cuda", non_blocking=True).unsqueeze(0) # [1, 124823, 3]
        
        
        # print("spherical_rt",spherical_rt.sum())
        cartesian_rt = self.sph_volume.sample_volume(spherical_rt, sample_points_spherical) # torch.Size([1, 2, 400, 400, 32])
        # print("cartesian_rt",cartesian_rt.sum())
        spherical_voxel_size = self.sph_volume.voxel_size(spherical_rt) # torch.Size([1, 3])
        cart_voxel_size = self.ego_volume.voxel_size(cartesian_rt) # torch.Size([1, 3])

        # there might be points containing "nan" values, since they have no valid transformation
        scale = (
            cart_voxel_size.prod(-1)[:, None, None, None]
            / spherical_volume_element_center_and_voxel_size(
                sample_points_spherical, spherical_voxel_size[:, None, None, None, :]
            )
        ).nan_to_num(0.0) # torch.Size([1, 400, 400, 32])
        cartesian_rt_scaled = scale.unsqueeze(1) * cartesian_rt # [1, 2, 400, 400, 32])
        self.agg_rt += cartesian_rt_scaled.sum(0, keepdim=True) # torch.Size([1, 2, 400, 400, 32])
        self.num_samples += cartesian_rt_scaled.shape[0] # 
        # print("self.num_samples",self.num_samples)


    def process_frames_in_radius(self, scene: pl.DataFrame):
        """Process frames in radius."""
        # 将半径在max_ego_pose_difference内的作为
        scene = scene.sort("LIDAR_TOP.sample_data.timestamp") # 场景列表 (382, 37)
        frame_ego_from_global = series_to_torch(scene["LIDAR_TOP.transform.ego_from_global"])  # torch.Size([382, 4, 4])  
        frame_ego_from_self_ego = frame_ego_from_global @ self.global_from_ego.cpu() # torch.Size([382, 4, 4])  
        pos_diff = frame_ego_from_self_ego[:, :3, 3].norm(dim=-1) # torch.Size([382]) 
        frams_in_radius = pos_diff < self.max_ego_pose_difference # torch.Size([382]) 
        scene = scene.with_columns(torch_to_series("frame_in_radius", frams_in_radius)) # (382, 38)
        scene = scene.filter(pl.col("frame_in_radius")) # (48, 38)
        if len(scene) > self.max_num_frames: # >50
            indices = torch.linspace(-0.5, len(scene) + 0.5, self.max_num_frames).round().long()
            scene = scene.with_row_count("index").filter(
                pl.col("index").is_in(torch_to_series("select_indices", indices))
            )
        items = scene.select(
            "LIDAR_TOP.transform.ego_from_global",
            "LIDAR_TOP.transform.sensor_from_ego",
            "scene.name",
            "LIDAR_TOP.sample_data.token",
        ).iter_slices(self.batch_size)
        # print("len",len(list(items))) # 49
        # 依次采样sample
        i = 0
        for sample in items:
            frame_dict = self._load_fn(sample) 
            # dict：frame_instance_ids，frame_instance_from_global，frame_ego_from_global，frame_lidar_from_ego， spherical_rt
            frame_dict = {k: v.to(**self.kwargs) for k, v in frame_dict.items()} 
            # dict：frame_instance_ids，frame_instance_from_global，frame_ego_from_global，frame_lidar_from_ego， spherical_rt
            # frame_instance_ids[1, 400, 400, 32], frame_instance_from_global[1, 228, 4, 4]), frame_ego_from_global([1, 4, 4]), 
            # frame_lidar_from_ego[1, 4, 4], spherical_rt[1, 2, 600, 100, 720]
            self._accumulate_frames(**frame_dict) # 每一个循环叠加一次历史帧信息

            


@dataclass
class TemporalTransmissionAndReflection:
    """Temporal accumulation of reflection and transmission using scene flow info."""

    ds: NuscenesDataset
    extra_data_root: Path
    reference_keyframes_only: bool = True
    frame_accumulation_kwargs: dict[str, Any] = field(default_factory=dict)
    missing_only: bool = False
    scene_offset: int = 0
    num_scenes: Optional[int] = None

    def save_accumulated_sample(self, scene_name: str, sample: pl.DataFrame):
        """Saves a single sample."""
        assert len(sample) == 1
        filename = self.save_path(scene_name, sample)
        filename.parent.mkdir(exist_ok=True, parents=True)
        # remove batch dim
        sample.write_ipc(filename, compression="zstd")

    def save_path(self, scene_name: str, sample: pl.DataFrame):
        """Save path."""
        assert len(sample) == 1
        filename = (
            Path(self.extra_data_root)
            / "reflection_and_transmission_multi_frame"
            / scene_name
            / "LIDAR_TOP"
            / f"{sample.item(0, 'LIDAR_TOP.sample_data.token')}.arrow"
        )
        return filename

    def process_scene(self, scene: pl.DataFrame):
        """Processes single scene."""
        scene = self.ds.join(scene, self.ds.sample)
        scene = self.ds.load_sample_data(scene, "LIDAR_TOP", with_data=False)
        scene = self.ds.sort_by_time(scene)

        if self.reference_keyframes_only:
            reference_frames = scene.filter(pl.col("LIDAR_TOP.sample_data.is_key_frame"))
        else:
            reference_frames = scene
        
        for reference_frame in tqdm.tqdm(reference_frames.iter_slices(1), total=len(reference_frames), position=1):
            if self.missing_only:
                filename = self.save_path(reference_frame["scene.name"].item(), reference_frame)
                if filename.exists():
                    continue
            reference_frame = self.ds.load_reflection_and_transmission_spherical(reference_frame, self.extra_data_root)
            reference_frame = self.ds.load_scene_flow_polars(reference_frame, self.extra_data_root) # 列表信息
            # 初始化
            agg_frames = AccumulateFrames(
                self.ds,
                ref_frame=reference_frame,
                extra_data_root=self.extra_data_root,
                **self.frame_accumulation_kwargs,
            )
            # print("reference_frame",reference_frame)
            '''数据处理'''
            agg_frames.process_frames_in_radius(scene)
            # print("agg_frames.agg_rt",np.shape(agg_frames.agg_rt)) # [1, 2, 400, 400, 32]
            assert agg_frames.agg_rt is not None, "No frames accumulated"
            agg_frames_mean = agg_frames.agg_rt / agg_frames.num_samples # torch.Size([1, 2, 400, 400, 32])  
            data = (
                reference_frame.select(
                    "LIDAR_TOP.sample_data.token",
                    "LIDAR_TOP.scene_flow.volume.lower",
                    "LIDAR_TOP.scene_flow.volume.upper",
                )
                .rename(
                    {
                        "LIDAR_TOP.scene_flow.volume.lower": "LIDAR_TOP.reflection_and_transmission_multi_frame.volume.lower",
                        "LIDAR_TOP.scene_flow.volume.upper": "LIDAR_TOP.reflection_and_transmission_multi_frame.volume.upper",
                    }
                )
                .with_columns(
                    torch_to_series("LIDAR_TOP.reflection_and_transmission_multi_frame", agg_frames_mean.cpu()) 
                    # (2, 400, 400, 32)
                )
            )
            # data_0 = agg_frames_mean[0][0].cpu()
            # data_1 = agg_frames_mean[0][1].cpu()
            # array_0 = data_0.numpy()
            # array_1 = data_1.numpy()
            # np.savez("/home/leon/Documents/Projects/evi-map/tools/tempor/array_0.npz", data=array_0)
            # np.savez("/home/leon/Documents/Projects/evi-map/tools/tempor/array_1.npz", data=array_1)


            self.save_accumulated_sample(reference_frame["scene.name"].item(), data)

    def process_data(self) -> None:
        """Process dataset."""
        # load all sample data with lidar token
        scene_to_process = self.ds.scene.slice(self.scene_offset, self.num_scenes)
        for scene in (tbar := tqdm.tqdm(scene_to_process.iter_slices(1), total=len(scene_to_process), position=0)):
            scene_name = scene["scene.name"].item()
            tbar.set_description_str(f"Processing {scene_name}")
            self.process_scene(scene)
