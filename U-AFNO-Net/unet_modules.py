import torch
from torch import nn
import torch.nn.functional as F

# =============== UNet 相关组件 ===============

class DoubleConv(nn.Module):
    """(convolution => [BN] => GELU) * 2 - 基于UNet的双卷积块"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """UNet下采样模块：MaxPool + DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """UNet上采样模块：Upsample + Skip Connection + DoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """UNet输出层"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# =============== 适配SimVP的UNet编码器解码器 ===============

class UNetEncoder(nn.Module):
    """基于UNet的编码器，输出多尺度特征用于skip connections"""

    def __init__(self, C_in, C_hid, bilinear=False):
        super(UNetEncoder, self).__init__()
        self.C_in = C_in
        self.C_hid = C_hid
        self.bilinear = bilinear

        # UNet编码器路径
        self.inc = DoubleConv(C_in, C_hid)
        self.down1 = Down(C_hid, C_hid * 2)
        self.down2 = Down(C_hid * 2, C_hid * 4)
        self.down3 = Down(C_hid * 4, C_hid * 8)

        factor = 2 if bilinear else 1
        self.down4 = Down(C_hid * 8, C_hid * 16 // factor)

    def forward(self, x):
        # 编码器前向传播，保存所有中间特征用于skip connections
        x1 = self.inc(x)  # C_hid
        x2 = self.down1(x1)  # C_hid * 2
        x3 = self.down2(x2)  # C_hid * 4
        x4 = self.down3(x3)  # C_hid * 8
        x5 = self.down4(x4)  # C_hid * 16 (or * 8 if bilinear)

        # 返回最深层特征和所有skip connection特征
        return x5, [x4, x3, x2, x1]


class UNetDecoder(nn.Module):
    """基于UNet的解码器，接收多个skip connections"""

    def __init__(self, C_hid, C_out, bilinear=False):
        super(UNetDecoder, self).__init__()
        self.C_hid = C_hid
        self.C_out = C_out
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        # UNet解码器路径
        self.up1 = Up(C_hid * 16, C_hid * 8 // factor, bilinear)
        self.up2 = Up(C_hid * 8, C_hid * 4 // factor, bilinear)
        self.up3 = Up(C_hid * 4, C_hid * 2 // factor, bilinear)
        self.up4 = Up(C_hid * 2, C_hid, bilinear)
        self.outc = OutConv(C_hid, C_out)

    def forward(self, hid, skip_connections):
        """
        Args:
            hid: 来自中间网络的特征 [B, C_hid*16, H, W]
            skip_connections: 来自编码器的skip connection特征列表 [x4, x3, x2, x1]
        """
        x4, x3, x2, x1 = skip_connections

        # UNet解码器前向传播，使用skip connections
        x = self.up1(hid, x4)  # 使用x4作为skip connection
        x = self.up2(x, x3)  # 使用x3作为skip connection
        x = self.up3(x, x2)  # 使用x2作为skip connection
        x = self.up4(x, x1)  # 使用x1作为skip connection

        # 输出层
        logits = self.outc(x)
        return logits


class UNetEncoderCompact(nn.Module):
    """紧凑版UNet编码器，适用于较小的特征图"""

    def __init__(self, C_in, C_hid, bilinear=False):
        super(UNetEncoderCompact, self).__init__()
        self.C_in = C_in
        self.C_hid = C_hid
        self.bilinear = bilinear

        # 更紧凑的编码器（只有3层下采样）
        self.inc = DoubleConv(C_in, C_hid)
        self.down1 = Down(C_hid, C_hid * 2)
        self.down2 = Down(C_hid * 2, C_hid * 4)

        factor = 2 if bilinear else 1
        self.down3 = Down(C_hid * 4, C_hid * 8 // factor)

    def forward(self, x):
        x1 = self.inc(x)  # C_hid
        x2 = self.down1(x1)  # C_hid * 2
        x3 = self.down2(x2)  # C_hid * 4
        x4 = self.down3(x3)  # C_hid * 8

        return x4, [x3, x2, x1]


class UNetDecoderCompact(nn.Module):
    """紧凑版UNet解码器"""

    def __init__(self, C_hid, C_out, bilinear=False):
        super(UNetDecoderCompact, self).__init__()
        self.C_hid = C_hid
        self.C_out = C_out
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        # 紧凑的解码器路径
        self.up1 = Up(C_hid * 8, C_hid * 4 // factor, bilinear)
        self.up2 = Up(C_hid * 4, C_hid * 2 // factor, bilinear)
        self.up3 = Up(C_hid * 2, C_hid, bilinear)
        self.outc = OutConv(C_hid, C_out)

    def forward(self, hid, skip_connections):
        x3, x2, x1 = skip_connections

        x = self.up1(hid, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        logits = self.outc(x)
        return logits