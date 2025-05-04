import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Standard convolution block with BatchNorm and activation."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()  # SiLU (Swish) activation used in YOLOv8

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with two convolutions and a skip connection."""
    def __init__(self, in_channels):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = ConvBlock(in_channels, mid_channels, kernel_size=1, padding=0)
        self.conv2 = ConvBlock(mid_channels, in_channels, kernel_size=3)
        
    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class CSPBlock(nn.Module):
    """
    Cross Stage Partial Block (CSP).
    Used in YOLOv8 backbone.
    """
    def __init__(self, in_channels, out_channels, num_blocks=1):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3)
        self.conv2 = ConvBlock(out_channels, out_channels//2, kernel_size=1, padding=0)
        self.conv3 = ConvBlock(out_channels, out_channels//2, kernel_size=1, padding=0)
        
        self.blocks = nn.Sequential(
            *[ResidualBlock(out_channels//2) for _ in range(num_blocks)]
        )
        
        self.conv4 = ConvBlock(out_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, x):
        x = self.conv1(x)
        route = self.conv2(x)
        x = self.conv3(x)
        x = self.blocks(x)
        x = torch.cat([x, route], dim=1)
        return self.conv4(x)


class SPPFBlock(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv8.
    """
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = ConvBlock(in_channels, mid_channels, kernel_size=1, padding=0)
        self.conv2 = ConvBlock(mid_channels * 4, out_channels, kernel_size=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        
    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        out = torch.cat([x, y1, y2, y3], dim=1)
        return self.conv2(out)


class UpsampleBlock(nn.Module):
    """Upsampling block for YOLOv8 decoder path."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, x):
        x = self.conv(x)
        return self.upsample(x)


class YOLOv8Seg(nn.Module):
    """
    YOLOv8 for Semantic Segmentation.
    Adapted for medical image segmentation tasks.
    """
    def __init__(self, in_channels=3, n_classes=1):
        super().__init__()
        
        # Backbone (simplified YOLOv8 backbone)
        # P1/2
        self.conv1 = ConvBlock(in_channels, 64, kernel_size=3, stride=2)
        # P2/4
        self.conv2 = ConvBlock(64, 128, kernel_size=3, stride=2)
        self.csp1 = CSPBlock(128, 128, num_blocks=1)
        # P3/8
        self.conv3 = ConvBlock(128, 256, kernel_size=3, stride=2)
        self.csp2 = CSPBlock(256, 256, num_blocks=2)
        # P4/16
        self.conv4 = ConvBlock(256, 512, kernel_size=3, stride=2)
        self.csp3 = CSPBlock(512, 512, num_blocks=3)
        # P5/32
        self.conv5 = ConvBlock(512, 1024, kernel_size=3, stride=2)
        self.csp4 = CSPBlock(1024, 1024, num_blocks=1)
        self.sppf = SPPFBlock(1024, 1024)
        
        # Neck (feature pyramid) 
        # Upsampling path
        self.up1 = UpsampleBlock(1024, 512)
        self.csp_up1 = CSPBlock(1024, 512, num_blocks=1)  # 512 + 512 after concat
        
        self.up2 = UpsampleBlock(512, 256)
        self.csp_up2 = CSPBlock(512, 256, num_blocks=1)   # 256 + 256 after concat
        
        self.up3 = UpsampleBlock(256, 128)
        self.csp_up3 = CSPBlock(256, 128, num_blocks=1)   # 128 + 128 after concat
        
        # Segmentation head
        self.up_final = nn.Sequential(
            UpsampleBlock(128, 64),  # 2x upsampling
            UpsampleBlock(64, 32),   # 2x upsampling
            ConvBlock(32, 32, kernel_size=3)
        )
        
        self.seg_head = nn.Conv2d(32, n_classes, kernel_size=1)
        
    def forward(self, x):
        # Backbone
        x1 = self.conv1(x)
        x2 = self.csp1(self.conv2(x1))
        x3 = self.csp2(self.conv3(x2))
        x4 = self.csp3(self.conv4(x3))
        x5 = self.sppf(self.csp4(self.conv5(x4)))
        
        # Neck - upsampling path with skip connections
        p5 = x5
        p4 = self.csp_up1(torch.cat([self.up1(p5), x4], dim=1))
        p3 = self.csp_up2(torch.cat([self.up2(p4), x3], dim=1))
        p2 = self.csp_up3(torch.cat([self.up3(p3), x2], dim=1))
        
        # Segmentation head
        x = self.up_final(p2)
        x = self.seg_head(x)
        
        return x