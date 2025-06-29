import torch
import torch.nn as nn
import torch.nn.functional as F

dir(nn.Module)
class GridSampler(nn.Module):
    def __init__(self, mode='bilinear', align_corners=True):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, image, grid):
        """
        input:
            image (torch.Tensor): input image，(N, C, H, W)
            grid (torch.Tensor): transformation，(N, H_out, W_out, 2)

        output:
            torch.Tensor: transformed image，(N, C, H_out, W_out)。
        """
        grid = grid.permute(0, 2, 3, 1)
        warped_image = F.grid_sample(image, grid, mode=self.mode, align_corners=self.align_corners)
        return warped_image

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        # print(x.shape)
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip_x):

        x = self.up(x)
        x = self.conv(x)
        x = torch.cat([skip_x, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, kernels=16, bilinear=True):
        super(UNet, self).__init__()

        # Initial convolutions for each input
        self.inc = DoubleConv(in_channels, kernels)

        # Downsampling
        self.down1 = Down(kernels, 2*kernels)
        self.down2 = Down(2*kernels, 4*kernels)
        self.down3 = Down(4*kernels, 8*kernels)

        # Upsampling

        self.up1 = Up(8*kernels, 4*kernels)
        self.up2 = Up(4*kernels, 2*kernels)
        self.up3 = Up(2*kernels, kernels)

        # Output layer
        self.outc = OutConv(kernels, out_channels)

    def forward(self, x_t, x_d):

        # cat
        x = torch.cat([x_t, x_d], dim=1)

        x = self.inc(x)

        # Downsampling
        x1 = self.down1(x)
        # print(x1.shape)
        x2 = self.down2(x1)
        # print(x2.shape)
        x3 = self.down3(x2)
        # print(x3.shape)

        # Upsampling
        x6 = self.up1(x3, x2)
        x7 = self.up2(x6, x1)
        x8 = self.up3(x7, x)

        # Output layer
        logits = self.outc(x8)
        x_coords = torch.linspace(1, 128, 128).unsqueeze(0).unsqueeze(0).repeat(x_t.size(0), 128, 1)
        y_coords = torch.linspace(1, 128, 128).unsqueeze(0).unsqueeze(2).repeat(x_t.size(0), 1, 128)
        device = logits.device   
        x_coords = x_coords.to(device)  
        y_coords = y_coords.to(device)
        grid = (logits.permute(0, 2, 3, 1) + torch.stack((x_coords, y_coords), dim=-1)-64.5)/63.5
        deformable = F.grid_sample(x_t, grid, mode='bilinear', align_corners=True)

        return logits, deformable 

class UNet_32(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, kernels=16, bilinear=True):
        super(UNet_32, self).__init__()

        # Initial convolutions for each input
        self.inc = DoubleConv(in_channels, kernels)

        # Downsampling
        self.down1 = Down(kernels, 2*kernels)
        self.down2 = Down(2*kernels, 4*kernels)
        self.down3 = Down(4*kernels, 8*kernels)

        # Upsampling

        self.up1 = Up(8*kernels, 4*kernels)
        self.up2 = Up(4*kernels, 2*kernels)
        self.up3 = Up(2*kernels, kernels)

        # Output layer
        self.outc = OutConv(kernels, out_channels)

    def forward(self, x_t, x_d):

        # cat
        x = torch.cat([x_t, x_d], dim=1)

        x = self.inc(x)

        # Downsampling
        x1 = self.down1(x)
        # print(x1.shape)
        x2 = self.down2(x1)
        # print(x2.shape)
        # print(x3.shape)

        # Upsampling
 
        x7 = self.up2(x2, x1)
        x8 = self.up3(x7, x)

        # Output layer
        logits = self.outc(x8)

        x_coords = torch.linspace(1, 128, 128).unsqueeze(0).unsqueeze(0).repeat(x_t.size(0), 128, 1)
        y_coords = torch.linspace(1, 128, 128).unsqueeze(0).unsqueeze(2).repeat(x_t.size(0), 1, 128)
        device = logits.device  
        x_coords = x_coords.to(device)  
        y_coords = y_coords.to(device)
        grid = (logits.permute(0, 2, 3, 1) + torch.stack((x_coords, y_coords), dim=-1)-64.5)/63.5
        deformable = F.grid_sample(x_t, grid, mode='bilinear', align_corners=True)

        return logits, deformable  

class UNet_8(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, kernels=16, bilinear=True):
        super(UNet_8, self).__init__()

        # Initial convolutions for each input
        self.inc = DoubleConv(in_channels, kernels)

        # Downsampling
        self.down1 = Down(kernels, 2*kernels)
        self.down2 = Down(2*kernels, 4*kernels)
        self.down3 = Down(4*kernels, 8*kernels)
        self.down4 = Down(8*kernels, 16*kernels)
        # Upsampling
        self.up0 = Up(16*kernels, 8*kernels)
        self.up1 = Up(8*kernels, 4*kernels)
        self.up2 = Up(4*kernels, 2*kernels)
        self.up3 = Up(2*kernels, kernels)

        # Output layer
        self.outc = OutConv(kernels, out_channels)

    def forward(self, x_t, x_d):

        # cat
        x = torch.cat([x_t, x_d], dim=1)

        x = self.inc(x)

        # Downsampling
        x1 = self.down1(x)
        # print(x1.shape)
        x2 = self.down2(x1)
        # print(x2.shape)
        x3 = self.down3(x2)
        # print(x3.shape)
        x4 = self.down4(x3)
        
        # Upsampling
        x5 = self.up0(x4, x3)
        x6 = self.up1(x3, x2)
        x7 = self.up2(x6, x1)
        x8 = self.up3(x7, x)

        # Output layer
        logits = self.outc(x8)
        x_coords = torch.linspace(1, 128, 128).unsqueeze(0).unsqueeze(0).repeat(x_t.size(0), 128, 1)
        y_coords = torch.linspace(1, 128, 128).unsqueeze(0).unsqueeze(2).repeat(x_t.size(0), 1, 128)
        device = logits.device 
        x_coords = x_coords.to(device) 
        y_coords = y_coords.to(device)
        grid = (logits.permute(0, 2, 3, 1) + torch.stack((x_coords, y_coords), dim=-1)-64.5)/63.5
        deformable = F.grid_sample(x_t, grid, mode='bilinear', align_corners=True)

        return logits, deformable  

class UNet_label(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, kernels=16, bilinear=True):
        super(UNet_label, self).__init__()

        # Initial convolutions for each input
        self.inc = DoubleConv(in_channels, kernels)

        # Downsampling
        self.down1 = Down(kernels, 2*kernels)
        self.down2 = Down(2*kernels, 4*kernels)
        self.down3 = Down(4*kernels, 8*kernels)

        # Upsampling

        self.up1 = Up(8*kernels, 4*kernels)
        self.up2 = Up(4*kernels, 2*kernels)
        self.up3 = Up(2*kernels, kernels)

        # Output layer
        self.outc = OutConv(kernels, out_channels)

    def forward(self, x_t, x_d, label):

        # cat
        x = torch.cat([x_t, x_d], dim=1)

        x = self.inc(x)

        # Downsampling
        x1 = self.down1(x)
        # print(x1.shape)
        x2 = self.down2(x1)
        # print(x2.shape)
        x3 = self.down3(x2)
        # print(x3.shape)

        # Upsampling
        x6 = self.up1(x3, x2)
        x7 = self.up2(x6, x1)
        x8 = self.up3(x7, x)

        # Output layer
        logits = self.outc(x8)

        x_coords = torch.linspace(1, 128, 128).unsqueeze(0).unsqueeze(0).repeat(x_t.size(0), 128, 1) 
        y_coords = torch.linspace(1, 128, 128).unsqueeze(0).unsqueeze(2).repeat(x_t.size(0), 1, 128)
        device = logits.device  
        x_coords = x_coords.to(device)  
        y_coords = y_coords.to(device)
        grid = (logits.permute(0, 2, 3, 1) + torch.stack((x_coords, y_coords), dim=-1)-64.5)/63.5   
        deformable = F.grid_sample(x_t, grid, mode='bilinear', align_corners=True)
        label_flow = F.grid_sample(label, grid, mode='bilinear', align_corners=True)

        return logits, deformable, label_flow 


