import torch
import numpy as np
import torch.nn as nn
import sys
# from torch.nn import Sequential as Seq, Linear as Lin, ReLU

def power_transform(x, gamma=2.0):
    x = x.float()
    x = x.clamp_min(0)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    x_normalized = x / (x_max + 1e-6)
    return x_normalized.pow(1.0 / gamma)

class Cross_Attention(torch.nn.Module):
    # [1, 5000, 8, 64] + [1, 5000, 8, 16] -> [1,5000, 8, 64]
    def __init__(self, embed_size_a = 64, embed_size_b = 16):
        super(Cross_Attention, self).__init__()
        self.scale = embed_size_a ** -0.5
        self.embed_size = 64

        self.query_v = nn.Linear(embed_size_a, embed_size_a, bias=False)
        # self.value_v = nn.Linear(embed_size_a, embed_size_a, bias=False)
        self.key_t = nn.Linear(embed_size_b, embed_size_a, bias=False)
        self.value_t = nn.Linear(embed_size_b, embed_size_a, bias=False)

        self.layer_norm = torch.nn.LayerNorm(embed_size_a, eps=1e-6)

        self.to_out = nn.Sequential(
            nn.Linear(embed_size_a, embed_size_a),
            nn.Dropout(0.1),
            nn.LayerNorm(embed_size_a, eps=1e-5)
        )

    def forward(self, voxel_code, trans_code):
        voxel_query = self.query_v(voxel_code) # [b, n, 8, 64]

        trans_value = self.value_t(trans_code) # [b, n, 8, 64]
        trans_key = self.key_t(trans_code) # [b, n, 8, 64]

        trans_attention = torch.einsum('bnid, bnjd -> bnij', voxel_query, trans_key) # [b, n, 8, 8]
        trans_attention = (trans_attention * self.scale).softmax(dim=-1) # [b, n, 8, 16] 归一化
        voxel_code_new = torch.einsum('bnij, bnjd -> bnid', trans_attention, trans_value) # [b, n, 8, 8] * [b, n, 8, 64]
        
        return self.to_out(voxel_code_new + voxel_code)

class Flatten(nn.Module):
    # [b, 1, 8] -> [b, 8]
    def forward(self, x):
        return x.view(x.size(0) * 1, -1)

class voxel_Attention(nn.Module):
    # [5000, 8, 64] -> [5000, 8, 64]
    def __init__(self, gate_channels=8, reduction_ratio=2):
        super(voxel_Attention, self).__init__()
        self.gate_channels = gate_channels
        self.ave_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.mlp_trans = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        b, n, c, l =  x.size() #[1, 50000, 8, 64]

        x = x.contiguous().view([b * n, c, l])
        avg_pool = self.ave_pool(x).view([b*n, 1, c]) # [b*n, 8, 1]->[b*n, 1, 8]
        max_pool = self.max_pool(x).view([b*n, 1, c]) # [b*n, 8, 1]->[b*n, 1, 8]

        voxel_ = y.contiguous().view([b*n, 1, c]) # [b*n, 1, 8]
        channel_att_trans = self.mlp_trans(voxel_)
        channel_att_avg = self.mlp(avg_pool)
        channel_att_max = self.mlp(max_pool)

        channel_att_voxel = channel_att_avg + channel_att_max + channel_att_trans # [b*n, c]
        scale = self.sigmoid(channel_att_voxel).contiguous().view([b*n, c, 1]) # [b*n, c, 1] is_contiguous()

        return x * scale # [b*n, c, 64]
    
class shuffle_voxel(nn.Module):
    # [5000, 8, 64]  -> [5000, 64]
    """
    input: 
        voxel特征[b, n, 8, 64]
        point透射特征[b, n, 8, 2]
    output:
        pillar特征[b, n, 64]
    """
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.code_reftrans = nn.Sequential(
            nn.Linear(2, 8, bias=False), 
            nn.ReLU(inplace=True),
            nn.Linear(8, 16, bias=False)
        )
        self.cross_att = Cross_Attention(embed_size_a = 64, embed_size_b = 16) 
        self.voxel_att = voxel_Attention(gate_channels = 8, reduction_ratio = 2)
        self.shuffle_conv = nn.Sequential(
            nn.Linear(in_channels, in_channels//2, bias=False),
            nn.LayerNorm(in_channels//2, eps=1e-5),
            nn.ReLU(),
            nn.Linear(in_channels//2, out_channels, bias=False),
            nn.LayerNorm(out_channels, eps=1e-5),
            nn.ReLU())

    def forward(self, voxel_feature, ref_trans):
        b, n, _, _ = voxel_feature.size()

        reftrans_code = self.code_reftrans(ref_trans) # -> [b, n, 8, 16]
        voxel_feature = self.cross_att(voxel_feature, reftrans_code) # -> [b, n, 8, 64]

        reftrans_feature = ref_trans.sum(dim=-1, keepdim=True) # [b, n, 8, 2] -> [b, n, 8, 1]
        voxel_feature = self.voxel_att(voxel_feature, reftrans_feature).contiguous().view(b, n, 8 * 64) #  [b*n , 8, 64] -> [b, n , 8, 64] 
               
        pillar_feature = self.shuffle_conv(voxel_feature) # [b, n , 8, 64] -> [b, n , 64]

        return pillar_feature

class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES # 64
        self.nx, self.ny, self.nz = self.model_cfg.VOXEL_DIM_NUM
        print("[PointPillarScatter] grid_size is:",grid_size,"nx, ny, nz is:",self.nx, self.ny, self.nz)
        # porcess visibility map
        self.conv_obser = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.01),
            nn.SiLU(inplace=True),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.01),
            nn.SiLU(inplace=True),
        )
        self.voxel_fusion = shuffle_voxel(in_channels=8 * 64, out_channels=64)
        # assert self.nz == 1
        # 需要在 data_processor.py 里修改 grid_size 大小 确保 self.nz == 1

    def forward(self, batch_dict, **kwargs):
        """
        input:
           pillar_features:(M ,64)
           coords:(M, 4) 第一维是batch_index 其余维度为zyx (batch_index,z,y,x)
        output:
           batch_spatial_features:(batch_size, 64, 500, 1000)
       """
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords'] # [N,64],[N,4]
        batch_size = coords[:, 0].max().int().item() + 1
        # 将转换成为伪图像的数据存在到该列表中
        batch_spatial_features = []  
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(   # [num_bev_features, nz * nx * ny]
                64,
                self.nz * self.nx * self.ny, # 500*1000*1
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx # 过滤出属于当前批次的voxel
            this_coords = coords[batch_mask, :]
            #计算索引, z * W * H + y * W + x
            # indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3] # 不考虑z轴
            indices = this_coords[:, 1] * self.ny * self.nx + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)

            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()

            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)
        
        batch_spatial_features = torch.stack(batch_spatial_features, 0) # list -> [1, 64, 500000]
        batch_spatial_features = batch_spatial_features.contiguous().view(batch_size,  # 1 
                                                                          self.num_bev_features, # 64
                                                                          self.nz, # 8
                                                                          self.ny, # 500
                                                                          self.nx) # 1000
        batch_features = batch_spatial_features.permute(0, 3, 4, 2, 1).contiguous().view(batch_size,  # [1, 500000, 8, 64]
                                                                                          self.ny*self.nx,
                                                                                          self.nz,
                                                                                          self.num_bev_features)
        # batch_flat_features = batch_features_.contiguous().view(batch_size*self.ny*self.nx, self.nz, self.num_bev_features) # [500000, 8, 64]
        ref_trans = torch.stack([batch_dict['reflections'], batch_dict['transmissions']], dim=1) #  [1, 2, 500, 1000, 8]
        ref_trans = ref_trans.permute(0, 2, 3, 4, 1).contiguous().view(batch_size, self.ny*self.nx, self.nz, 2) # [1, 500000, 8, 2]
        
        batch_pillar_features = self.voxel_fusion(batch_features, ref_trans) # [1, 500000, 64]
        batch_pillar_features = batch_pillar_features.permute(0, 2, 1).contiguous().view(batch_size, self.num_bev_features, self.ny, self.nx)

        observations = batch_dict['observations'].contiguous().view(batch_size, 1, self.ny, self.nx) # [1, 1, 500, 1000]

        binarymap = batch_dict['binarymap'].contiguous().view(batch_size, 1, self.ny, self.nx)

        observations_features = self.conv_obser(torch.cat((observations, binarymap), dim=1)) # [1, 64, 500, 1000]
        spatial_features = torch.cat([batch_pillar_features, observations_features], dim=1) # [1, 64, 500, 1000]+[1, 64, 500, 1000] -> [1, 128, 500, 1000]
        # spatial_features = batch_pillar_features + observations_features # [1, 64, 500, 1000] + [1, 64, 500, 1000] -> [1, 64, 500, 1000]
        batch_dict['spatial_features'] = spatial_features.contiguous()
        batch_dict['obser_features'] = observations_features.contiguous()
        
        return batch_dict
