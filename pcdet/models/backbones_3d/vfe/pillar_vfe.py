import torch

from ....ops.roiaware_pool3d import roiaware_pool3d_utils
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .vfe_template import VFETemplate

import sys
from torch_geometric.nn import FeaStConv
# from knn_cuda import KNN
from torch_cluster import fps

class PFNLayer(nn.Module):
    '''
    简化版pointnet
    (M, 20, 10)->(M, 20, 32)->(M, 20, 64)
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        # BatchNorm1d在通道维度上进行,对于图像来说默认模式为[N,C,H*W],通道在第二个维度上,所以需要变换维度
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True

        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]  # 在维度1(柱状空间的点)使用最大池化操作，找出每个pillar中最能代表该pillar的点

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated
        
        
# 构建vfe模块 2024
class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)
        num_point_features = 4
        self.use_norm = self.model_cfg.USE_NORM # True
        self.with_distance = self.model_cfg.WITH_DISTANCE # False
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ # True
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS #[64]
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters) # [10, 64]
        # 构建一个简单的pointnet
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.relu = nn.ReLU()

        self.voxel_x = voxel_size[0] # 0.1
        self.voxel_y = voxel_size[1] # 0.1
        self.voxel_z = voxel_size[2] # 0.5 = 4/8
        # print("voxel_size",voxel_size)
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0] # 0.1/2 + -50 = -49.95
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1] # 0.1/2 + (-25) = -24.95
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2] # 40.5/2 + (-2.5) = -2.25


    def get_output_feature_dim(self):
        return self.num_filters[-1]
    

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        """
        batch_dict:
            points:(97687,5)
            frame_id:(4,) --> (2238,2148,673,593)
            gt_boxes:(4,40,8)--> (x,y,z,dx,dy,dz,ry,class)
            use_lead_xyz:(4,) --> (1,1,1,1)
            voxels:(31530,32,4) --> (x,y,z,intensity)
            voxel_coords:(31530,4) --> (batch_index,z,y,x) 在dataset.collate_batch中增加了batch索引
            voxel_num_points:(31530,)
            image_shape:(4,2) [[375 1242],[374 1238],[375 1242],[375 1242]]
            batch_size:4
        """
        # 1、获取voxel特征
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], \
                                                    batch_dict['voxel_num_points'], \
                                                    batch_dict['voxel_coords']  
        #   voxels 当前点云中包含的所有点 [30000, 20, 5],
        #   voxel_num_points 体素内有多少实际有效的[30000], 
        #   voxels_coord每个体素坐标[30000, 4]
        coords = batch_dict['voxel_coords']
        batch_size = coords[:, 0].max().int().item() + 1 # 1
        voxel_features= voxel_features[:,:,:4] # [30000, 20, 4] 最后一层全0不需要

        # 2、计算voxle中，每个点相对于平均值的偏移量
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1) # [30000, 1, 3]
        f_cluster = voxel_features[:, :, :3] - points_mean # [30000, 20, 3]
        f_center = torch.zeros_like(voxel_features[:,: , :3]) #[30000, 20, 3]
        center = torch.zeros_like(voxel_features[:,1,:3]).view(voxel_features.size()[0],1,3) # [30000, 1, 3]

        # 3、网格点coord乘每个voxl的长宽得到实际长宽，加上每个voxl长宽的一半得到中心点坐标 ps:center好像没用到
        center[:,:,0] = (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        center[:,:,1] = (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        center[:,:,2] = (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        # 4、每个点的x、y、z减去对应voxl的中心点，得到每个点到该点中心点的偏移量
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
        # 扩充信息(x,y,z,xc,yc,zc,xp,yp,zp)
        if self.use_absolute_xyz: # True 绝对坐标x,y,z+和平均值的偏差xc,yc,zc+中心偏移量xp,yp,zp
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance: # False
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        #   按最后一维拼接
        features = torch.cat(features, dim=-1) # [30000, 20, 10=4+3+3]

        # 5、生成每个voxel中，不满足最大20个点的voxel会存在由0填充的数据，填充的数据在计算出现xc,yc,zc和xp,yp,zp出现数值，用get_paddings_indicator清零
        voxel_count = features.shape[1] # 20个point
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0) # [30000, 20]
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features) # (30000,20)->(30000,20,1)
        features *= mask # [30000, 20, 10]

        for pfn in self.pfn_layers:
            features = pfn(features)

        features = features.squeeze() # [30000, 1, 64] -> [30000, 64]
        batch_dict['pillar_features'] = features
        return batch_dict

