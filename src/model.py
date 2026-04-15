import torch
import torch.nn as nn
import torch.nn.functional as F

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

class SpectrogramUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        """
        U-Net architecture for Vocal Separation.
        Inputs: Magnitude Spectrogram of mixed track (1 channel)
        Outputs: Mask for Vocal Magnitude Spectrogram (1 channel)
        """
        super().__init__()
        self.inc = UNetConvBlock(n_channels, 32)
        
        # Encoder
        self.down1 = nn.Sequential(nn.MaxPool2d(2), UNetConvBlock(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), UNetConvBlock(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), UNetConvBlock(128, 256))
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = UNetConvBlock(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = UNetConvBlock(128, 64)
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = UNetConvBlock(64, 32)
        
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Downsample
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Upsample 1
        u1 = self.up1(x4)
        diffY = x3.size()[2] - u1.size()[2]
        diffX = x3.size()[3] - u1.size()[3]
        u1 = F.pad(u1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x_up1 = self.conv_up1(torch.cat([x3, u1], dim=1))
        
        # Upsample 2
        u2 = self.up2(x_up1)
        diffY = x2.size()[2] - u2.size()[2]
        diffX = x2.size()[3] - u2.size()[3]
        u2 = F.pad(u2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x_up2 = self.conv_up2(torch.cat([x2, u2], dim=1))
        
        # Upsample 3
        u3 = self.up3(x_up2)
        diffY = x1.size()[2] - u3.size()[2]
        diffX = x1.size()[3] - u3.size()[3]
        u3 = F.pad(u3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x_up3 = self.conv_up3(torch.cat([x1, u3], dim=1))
        
        # Output Mask (Values between 0 and 1)
        mask = self.sigmoid(self.outc(x_up3))
        return mask
