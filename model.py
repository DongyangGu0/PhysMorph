import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter

class DoubleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SelfAttention3D(nn.Module):
    
    def __init__(self, in_dim):
        super(SelfAttention3D, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  

    def forward(self, x):
        m_batchsize, C, depth, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, depth * width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, depth * width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(m_batchsize, -1, depth * width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, depth, width, height)
        out = self.gamma * out + x
        return out

def interp3(prior, dvfx, dvfy, dvfz):
    """
    3D interp func

    Args:
        prior: [B, 1, H, W, D] #img
        dvfx, dvfy, dvfz: [B, H, W, D] #dvf in x,y,z

    Returns:
        [B, 1, H, W, D] #deformed img
    """
    B, C, H, W, D = prior.shape

    y, x, z = torch.meshgrid(
        torch.arange(H, device=prior.device),
        torch.arange(W, device=prior.device),
        torch.arange(D, device=prior.device),
        indexing='ij'
    )

    new_x = x.unsqueeze(0) + dvfx
    new_y = y.unsqueeze(0) + dvfy
    new_z = z.unsqueeze(0) + dvfz

    x0 = torch.floor(new_x).long()
    x1 = x0 + 1
    y0 = torch.floor(new_y).long()
    y1 = y0 + 1
    z0 = torch.floor(new_z).long()
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, W - 1)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 1)
    y1 = torch.clamp(y1, 0, H - 1)
    z0 = torch.clamp(z0, 0, D - 1)
    z1 = torch.clamp(z1, 0, D - 1)

    xd = (new_x - x0.float()).unsqueeze(1)
    yd = (new_y - y0.float()).unsqueeze(1)
    zd = (new_z - z0.float()).unsqueeze(1)

    c000 = prior[:, :, y0, x0, z0]
    c001 = prior[:, :, y0, x0, z1]
    c010 = prior[:, :, y0, x1, z0]
    c011 = prior[:, :, y0, x1, z1]
    c100 = prior[:, :, y1, x0, z0]
    c101 = prior[:, :, y1, x0, z1]
    c110 = prior[:, :, y1, x1, z0]
    c111 = prior[:, :, y1, x1, z1]

    c00 = c000 * (1 - zd) + c001 * zd
    c01 = c010 * (1 - zd) + c011 * zd
    c10 = c100 * (1 - zd) + c101 * zd
    c11 = c110 * (1 - zd) + c111 * zd

    c0 = c00 * (1 - xd) + c01 * xd
    c1 = c10 * (1 - xd) + c11 * xd

    c = c0 * (1 - yd) + c1 * yd

    return c

class SpatialTransformationLayer(nn.Module):
    def __init__(self):
        super(SpatialTransformationLayer, self).__init__()

    def forward(self, image, dvf):
        # DVF in x,y,z
        dvfy = dvf[:, 0]  # 1st: y
        dvfx = dvf[:, 1]  # 2nd: x
        dvfz = dvf[:, 2]  # 3rd: z
        transformed_image = interp3(image, dvfx, dvfy, dvfz)
        return transformed_image

def fill_holes(mask):
    """
    mask: [B, 1, H, W, D]
    """
    mask_np = mask.cpu().numpy()  # numpy
    filled_mask = np.zeros_like(mask_np)
    for i in range(mask_np.shape[0]):
        # binary_fill_holes: bool needed
        filled = binary_fill_holes(mask_np[i, 0].astype(bool))
        filled_mask[i, 0] = filled.astype(np.float32)
    return torch.from_numpy(filled_mask).to(mask.device)

def smooth_mask(mask, sigma=1):
    """
    mask: [B, 1, H, W, D]。
    """
    mask_np = mask.cpu().numpy()
    smoothed_mask = np.zeros_like(mask_np)
    for i in range(mask_np.shape[0]):
        # Smooth processing with Gaussian filtering
        smooth = gaussian_filter(mask_np[i, 0], sigma=sigma)
        # 0.5 threshold 
        smoothed_mask[i, 0] = (smooth >= 0.5).astype(np.float32)
    return torch.from_numpy(smoothed_mask).to(mask.device)

class UNet3D(nn.Module):
    """3D U-Net"""
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(UNet3D, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck with Self-Attention
        self.bottleneck = nn.Sequential(
            DoubleConv(features[-1], features[-1] * 2),
            SelfAttention3D(features[-1] * 2)
        )
        
        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature * 2, feature))

        # Output: 3 channel DVF
        self.final_conv = nn.Conv3d(features[0], 3, kernel_size=1)
        
        # Spatial transform layer
        self.spatial_transform = SpatialTransformationLayer()

    def forward(self, x):

        cbct_image = x[:, 0:1, ...]
        mr_image = x[:, 1:2, ...]

        skip_connections = []

        # Encoder
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck with self-attention
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)

        # Generate DVF
        dvf = self.final_conv(x)
        
        # Apply DVF to get MR
        transformed_image = self.spatial_transform(mr_image, dvf)
        
        # Change Dim to [B, H, W, D, 3]
        dvf = dvf.permute(0, 2, 3, 4, 1)
        
        # Cal mask
        mask = (torch.abs(cbct_image) >= config.liver_mask_threshold).float()
        mask = fill_holes(mask)
        mask = smooth_mask(mask, sigma=1)  # sigma adjustable

        # Apply mask on deformed mr
        transformed_image = transformed_image * mask
        
        # mask dim needs to fit dim of MR：[B, 1, H, W, D] -> [B, H, W, D, 1]
        mask_dvf = mask.squeeze(1).unsqueeze(-1)
        dvf = dvf * mask_dvf
        
        return transformed_image, dvf