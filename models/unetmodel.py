import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Double Convolution block: (Conv2d -> BatchNorm -> ReLU) x 2
    
    This block is used multiple times in the U-Net architecture.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        
        # TODO: Implement the double convolution block
        # Hint: Create a sequential module with two Conv2d layers
        # Each followed by BatchNorm2d and ReLU
        # The first Conv2d should go from in_channels to out_channels
        # The second should maintain the out_channels
        
        self.double_conv = nn.Sequential(
            # Your code here
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # TODO: Implement the forward pass of the double convolution block
        # Hint: Apply the double_conv sequential to the input tensor
        
        return self.double_conv(x) # Replace with your implementation


class DownSample(nn.Module):
    """
    Downsampling block: MaxPool2d followed by DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        
        # TODO: Implement the downsampling block
        # Hint: Use MaxPool2d with kernel_size=2 for downsampling
        # Then use the DoubleConv block you implemented above
        
        self.maxpool_conv = nn.Sequential(
            # Your code here
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        # TODO: Implement the forward pass of the downsampling block
        # Hint: Apply the maxpool_conv sequential to the input tensor
        
        return self.maxpool_conv(x)  # Replace with your implementation


class UpSample(nn.Module):
    """
    Upsampling block: Upsample followed by DoubleConv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpSample, self).__init__()
        
        # TODO: Implement the upsampling block
        # Hint: If bilinear is True, use nn.Upsample with mode='bilinear'
        # If bilinear is False, use ConvTranspose2d for learnable upsampling
        # Then reduce the number of channels by a DoubleConv
        
        if bilinear:
            # Your code here for bilinear upsampling
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
            pass
        else:
            # Your code here for transposed convolution upsampling
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            pass
            
    def forward(self, x1, x2):
        # TODO: Implement the forward pass of the upsampling block
        # Hint: First upsample x1, then handle the concatenation with x2
        # Consider that x1 and x2 may have different spatial dimensions
        # You'll need to deal with this using operations like center crop
        # Then apply the double convolution
        
        # Your code here
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)  # Replace with your implementation


class OutConv(nn.Module):
    """
    Output Convolution block: 1x1 convolution to map to the required number of classes
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        
        # TODO: Implement the output convolution
        # Hint: Use a 1x1 Conv2d to map to the required number of output classes
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # Replace with your implementation
    
    def forward(self, x):
        # TODO: Implement the forward pass of the output convolution
        # Hint: Apply the conv layer to the input tensor
        
        return self.conv(x) # Replace with your implementation


class UNet(nn.Module):
    """
    Full U-Net architecture
    """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()

        base_features = 64
        
        # Input layer
        # TODO: Implement the input double convolution
        # Hint: Use DoubleConv to process the input image
        self.inc = DoubleConv(n_channels, base_features)  # Replace with your implementation
        
        # Encoder (downsampling) path
        # TODO: Implement the encoder path with multiple downsampling steps
        # Hint: Create multiple DownSample modules with increasing channel depth
        # Typical channel progression: 64 -> 128 -> 256 -> 512 -> 1024
        self.down1 = DownSample(base_features, base_features * 2)  # Replace with your implementation
        self.down2 = DownSample(base_features * 2, base_features * 4)  # Replace with your implementation
        self.down3 = DownSample(base_features * 4, base_features * 8)  # Replace with your implementation
        factor = 2 if bilinear else 1
        self.down4 = DownSample(base_features * 8, base_features * 16 // factor)  # Replace with your implementation
        
        # Bottleneck
        # This is handled by the last downsampling step
        
        # Decoder (upsampling) path
        # TODO: Implement the decoder path with multiple upsampling steps
        # Hint: Create multiple UpSample modules with decreasing channel depth
        # Typical channel progression: 1024 -> 512 -> 256 -> 128 -> 64
        self.up1 = UpSample(base_features * 16, base_features * 8 // factor, bilinear)  # Replace with your implementation
        self.up2 = UpSample(base_features * 8, base_features * 4 // factor, bilinear)  # Replace with your implementation
        self.up3 = UpSample(base_features * 4, base_features * 2 // factor, bilinear)  # Replace with your implementation
        self.up4 = UpSample(base_features * 2, base_features, bilinear)  # Replace with your implementation
        
        # Output layer
        # TODO: Implement the output convolution
        # Hint: Use OutConv to produce the final segmentation map
        self.outc = OutConv(base_features, n_classes)  # Replace with your implementation
    
    def forward(self, x):
        # TODO: Implement the forward pass of the U-Net
        # Hint: Follow the U-Net architecture diagram
        # 1. Apply the input convolution
        # 2. Apply encoder blocks and save the outputs for skip connections
        # 3. Apply decoder blocks with skip connections
        # 4. Apply the output convolution
        
        # Input convolution
        # Your code here
        x1 = self.inc(x)
        
        # Encoder path
        # Your code here
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path with skip connections
        # Your code here
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output convolution
        # Your code here
        logits = self.outc(x)
        
        return logits  # Replace with your implementation