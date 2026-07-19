"""
Feature 5C — Texture Feature Extraction
======================================
Extracts edge representations (Laplacian) and Local Binary Pattern (LBP) micro-textures,
projecting these descriptors through a dedicated convolutional network branch.
"""

from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


class TextureEncoder(nn.Module):
    """
    Lightweight CNN to encode multi-channel texture/edge maps.
    Input size: (Batch, 3, 112, 112) BGR -> Output size: (Batch, embedding_dim).
      Channel 0: Laplacian edge map (high-frequency detail)
      Channel 1: Sobel gradient magnitude / LBP representation
    """

    def __init__(self, embedding_dim: int = 512) -> None:
        super().__init__()
        
        # Fixed weights for Sobel and Laplacian filters
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        laplacian = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('laplacian', laplacian)
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),  # -> 16x56x56
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> 32x28x28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> 64x14x14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# -> 128x7x7
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))                           # -> 128x1x1
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: BGR face crop tensor of shape (Batch, 3, Height, Width), normalized to [0, 1]
        """
        # Convert BGR to Grayscale
        gray = 0.114 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.299 * x[:, 2:3, :, :]
        
        # Apply Laplacian
        lap = nn.functional.conv2d(gray, self.laplacian, padding=1)
        lap = torch.abs(lap)
        
        # Apply Sobel
        sx = nn.functional.conv2d(gray, self.sobel_x, padding=1)
        sy = nn.functional.conv2d(gray, self.sobel_y, padding=1)
        sob = torch.sqrt(sx**2 + sy**2 + 1e-6)
        
        # Normalize to [0, 1]
        lap_max = lap.view(lap.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1) + 1e-6
        lap = lap / lap_max
        sob_max = sob.view(sob.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1) + 1e-6
        sob = sob / sob_max
        
        texture_maps = torch.cat([lap, sob], dim=1) # (B, 2, H, W)
        
        features = self.conv(texture_maps)
        embeddings = self.fc(features)
        # L2 normalize texture embeddings
        return nn.functional.normalize(embeddings, p=2, dim=1)


class TextureFeatureExtractor:
    """
    Helper to extract edge and texture maps from aligned face crops.
    """

    @staticmethod
    def extract_texture_maps(img: np.ndarray) -> np.ndarray:
        """
        Returns the BGR image as is, since texture map extraction is now
        performed directly in the TextureEncoder on the GPU.
        """
        return img

    @staticmethod
    def preprocess_texture(texture_map: np.ndarray) -> torch.Tensor:
        """Convert numpy BGR image (H, W, 3) to PyTorch tensor (1, 3, H, W)."""
        tensor = torch.from_numpy(texture_map.transpose(2, 0, 1)).float() / 255.0
        return tensor.unsqueeze(0)

