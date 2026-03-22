import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SwinModel

class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(len(self.in_channels) - 1):
            l_conv = nn.Sequential(
                nn.Conv2d(self.in_channels[i] + (self.in_channels[i + 1] if i == len(self.in_channels) - 2 else self.out_channels), self.out_channels, 1),
                nn.BatchNorm2d(self.out_channels, eps=1e-5, momentum=0.1),
                nn.ReLU(inplace=True),   
            )
            f_conv = nn.Sequential(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(self.out_channels, eps=1e-5, momentum=0.1),
                nn.ReLU(inplace=True),
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(f_conv)

    def forward(self, inputs):
        """
        Args:
            inputs: list of feature maps, from high to low resolution e.g. [(H, W), (H/2, W/2), (H/4, W/4)]
        Returns:
            outs: list of feature maps, from high to low resolution e.g. [(H, W), (H/2, W/2)]
        """
        laterals = inputs
        for i in range(len(inputs) - 2, -1, -1):
            x = F.interpolate(laterals[i + 1], laterals[i].shape[-2:], mode='bilinear', align_corners=False)
            laterals[i] = torch.cat([laterals[i], x], 1)
            laterals[i] = self.lateral_convs[i](laterals[i])
            laterals[i] = self.fpn_convs[i](laterals[i])
        
        outs = [laterals[i] for i in range(len(laterals) - 1)]
        return outs
    
class SwinT_tiny_Encoder(nn.Module):
    def __init__(self, model_cfg):
        '''
        output_indices: [1, 2, 3] 表示输出第1, 2, 3层特征
        featureShape: (256, 32, 88)
        out_channels: 256
        FPN_in_channels: [192, 384, 768]
        FPN_out_channels: 256
        '''
        super(SwinT_tiny_Encoder, self).__init__()
        self.model_cfg = model_cfg 
        self.model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.output_indices = self.model_cfg.get('OUT_INDICES', [1, 2, 3])
        FPN_in_channels = self.model_cfg.get('FPN_IN_CHANNELS', [192, 384, 768])
        FPN_out_channels = self.model_cfg.get('FPN_OUT_CHANNELS', 256)
        self.FPN = FPN(FPN_in_channels, FPN_out_channels)
        _, self.fH, self.fW = self.model_cfg.get('FEATURE_SHAPE', [256, 32, 88])
        self.out_channels = self.model_cfg.get('OUT_CHANNELS', 256)
        self.in_channels = self.model_cfg.get('IN_CHANNELS', 3)

    def forward(self, batch_dict):
        """
        Args:
            x: (B, N, C, H, W), N is the number of images at the same time
        Returns:
            out: (B, N, out_channels, fH, fW), feature maps
        使用了单张图，在调用前需要img = img.unsqueeze(1) # 变为 (B, 1, 3, H, W)，这里 N=1
        """
        imgs = batch_dict['images']
        # upsample -> cat -> conv1x1 -> conv3x3

        B, C, H, W = imgs.shape # [1, 3, 320, 1024]
        N = 1
        imgs = imgs.view(B * N, C, H, W)
        output = self.model(imgs, output_hidden_states=True)

        # for i in range(len(output.reshaped_hidden_states)):
        #     print("[DEBUG] output.reshaped_hidden_states[%d].shape: " % i, output.reshaped_hidden_states[i].shape)
        # [1, 96, 80, 256]，[1, 192, 40, 128]，[1, 384, 20, 64]，[1, 768, 10, 32]，[1, 768, 10, 32]
        ret = [output.reshaped_hidden_states[i] for i in self.output_indices] # 取[1, 2, 3] 层特征
        out = self.FPN(ret) # (B*N, 256, H/8, W/8) + (B*N, 256, H/16, W/16) + (B*N, 256, H/32, W/32) -> (B*N, 256, H/8, W/8) + (B*N, 256, H/16, W/16)
        batch_dict['image_fpn'] = tuple(out)

        return batch_dict