import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    标准的 Pre-activation 或 Post-activation ResNet Block 变体
    结构: Conv3x3 -> BN -> SiLU -> Conv3x3 -> BN (+ x) -> SiLU
    """
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        
        # 如果输入输出通道不一致，需要在残差连接上做 1x1 卷积投影
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.SiLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.act2(out)
        return out

class UpBlock(nn.Module):
    """
    上采样模块：Upsample -> Conv3x3 (Align) -> Concat -> ResBlock
    """
    def __init__(self, up_in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        
        if bilinear:
            # 使用 Upsample + 3x3 卷积将上层特征通道数对齐到 skip_channels
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(up_in_channels, skip_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(skip_channels),
                nn.SiLU(inplace=True)
            )
        else:
            self.up = nn.ConvTranspose2d(up_in_channels, skip_channels, kernel_size=2, stride=2)
            
        # 拼接后的通道数处理：(skip_channels + skip_channels) -> out_channels
        self.conv = ResBlock(skip_channels * 2, out_channels)

    @staticmethod
    def _pad_or_crop_to_match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        diff_y = ref.size(2) - x.size(2)
        diff_x = ref.size(3) - x.size(3)

        # pad（x 小）
        if diff_x > 0 or diff_y > 0:
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )

        # crop（x 大）
        if diff_x < 0 or diff_y < 0:
            crop_top = (-diff_y) // 2
            crop_left = (-diff_x) // 2
            x = x[:, :, crop_top:crop_top + ref.size(2), crop_left:crop_left + ref.size(3)]

        return x

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        x1 = self._pad_or_crop_to_match(x1, x2)
        # x1 是上采样并对齐通道后的特征，x2 是 Skip Connection 的特征
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ResUnet_64(nn.Module):
    def __init__(self, n_channels=64, n_classes=12, bilinear=True):
        super(ResUnet_64, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # 动态定义通道数配置
        # Level: Input -> 1 -> 2 -> 3 -> 4 -> Bottleneck
        # 修改为每一层下采样都增加通道数 (Input=64 -> 64 -> 128 -> 256 -> 512 -> 1024)
        self.filters = [n_channels, 64, 128, 256, 512, 1024]

        # Encoder
        # 保持与原网络类似的层级深度，但结构更标准
        self.inc = ResBlock(self.filters[0], self.filters[1])
        
        # Downsampling is done via MaxPool separately
        self.pool = nn.MaxPool2d(2)
        
        self.down1 = ResBlock(self.filters[1], self.filters[2])
        self.down2 = ResBlock(self.filters[2], self.filters[3])
        self.down3 = ResBlock(self.filters[3], self.filters[4])
        self.down4 = ResBlock(self.filters[4], self.filters[5])
        
        self.bridge = ResBlock(self.filters[5], self.filters[5])

        # Decoder
        # 上采样时输入通道 = 上一层的输出 + Skip Connection 的通道
        self.up4 = UpBlock(self.filters[5], self.filters[4], self.filters[4], bilinear)
        self.up3 = UpBlock(self.filters[4], self.filters[3], self.filters[3], bilinear)
        self.up2 = UpBlock(self.filters[3], self.filters[2], self.filters[2], bilinear)
        self.up1 = UpBlock(self.filters[2], self.filters[1], self.filters[1], bilinear)

        self.class_layer = nn.Sequential(
            nn.Conv2d(self.filters[1], 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.filters[1], momentum=0.1),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.filters[1], 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.filters[1], momentum=0.1),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.filters[1], n_classes, kernel_size=1, stride=1, padding=0),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder Path
        x1 = self.inc(x)      # Level 1 [B, 128, H, W]
        
        p1 = self.pool(x1)
        x2 = self.down1(p1)   # Level 2 [B, 128, H/2, W/2]
        
        p2 = self.pool(x2)
        x3 = self.down2(p2)   # Level 3 [B, 256, H/4, W/4]
        
        p3 = self.pool(x3)
        x4 = self.down3(p3)   # Level 4 [B, 512, H/8, W/8]
        
        p4 = self.pool(x4)
        x5 = self.down4(p4)   # Bottleneck Input [B, 1024, H/16, W/16]
        
        x5 = self.bridge(x5)  # Bottleneck Refine

        # Decoder Path
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        
        # Classification
        logits = self.class_layer(x)
        return logits

class ResUnet_128(nn.Module):
    def __init__(self, n_channels=128, n_classes=12, bilinear=True):
        super(ResUnet_128, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        

        self.filters = [n_channels, 128, 256, 512, 1024]

        # Encoder
        # 保持与原网络类似的层级深度，但结构更标准
        self.inc = ResBlock(self.filters[0], self.filters[1])
        
        # Downsampling is done via MaxPool separately
        self.pool = nn.MaxPool2d(2)
        
        self.down1 = ResBlock(self.filters[1], self.filters[1])
        self.down2 = ResBlock(self.filters[1], self.filters[2])
        self.down3 = ResBlock(self.filters[2], self.filters[3])
        self.down4 = ResBlock(self.filters[3], self.filters[4])
        
        self.bridge = ResBlock(self.filters[4], self.filters[4])

        # Decoder
        # 上采样时输入通道 = 上一层的输出 + Skip Connection 的通道
        self.up4 = UpBlock(self.filters[4], self.filters[3], self.filters[3], bilinear)
        self.up3 = UpBlock(self.filters[3], self.filters[2], self.filters[2], bilinear)
        self.up2 = UpBlock(self.filters[2], self.filters[1], self.filters[1], bilinear)
        self.up1 = UpBlock(self.filters[1], self.filters[1], self.filters[1], bilinear)

        self.class_layer = nn.Sequential(
            nn.Conv2d(self.filters[1], 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder Path
        x1 = self.inc(x)      # Level 1 [B, 128, H, W]
        
        p1 = self.pool(x1)
        x2 = self.down1(p1)   # Level 2 [B, 128, H/2, W/2]
        
        p2 = self.pool(x2)
        x3 = self.down2(p2)   # Level 3 [B, 256, H/4, W/4]
        
        p3 = self.pool(x3)
        x4 = self.down3(p3)   # Level 4 [B, 512, H/8, W/8]
        
        p4 = self.pool(x4)
        x5 = self.down4(p4)   # Bottleneck Input [B, 1024, H/16, W/16]
        
        x5 = self.bridge(x5)  # Bottleneck Refine

        # Decoder Path
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        
        # Classification
        logits = self.class_layer(x)
        return logits
        
        



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResUnet(n_channels=128, n_classes=12).to(device)

    # 模拟输入
    input = torch.randn(1, 128, 500, 1000).to(device)
    out = model(input)
    print("Output size:", out.size()) # Should be [1, 12, 500, 1000]

    total = sum([param.nelement() for param in model.parameters()])
    print("Total parameters:", total)