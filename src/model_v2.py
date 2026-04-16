import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, channels, height, width = x.size()
        proj_query = self.query_conv(x).view(batch, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        out = self.gamma * out + x
        return out

class UNetConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class SpectrogramUNetv2(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        self.inc = UNetConvBlock(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), UNetConvBlock(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), UNetConvBlock(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), UNetConvBlock(128, 256))
        
        # [M2 核心升級] 在最底層 Bottleneck 加入自注意力
        self.attention = SelfAttention(256)
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = UNetConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = UNetConvBlock(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = UNetConvBlock(64, 32)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # 套用注意力
        x4 = self.attention(x4)
        
        u1 = self.up1(x4)
        x_up1 = self.conv_up1(torch.cat([x3, u1], dim=1))
        u2 = self.up2(x_up1)
        x_up2 = self.conv_up2(torch.cat([x2, u2], dim=1))
        u3 = self.up3(x_up2)
        x_up3 = self.conv_up3(torch.cat([x1, u3], dim=1))
        
        return self.sigmoid(self.outc(x_up3))
