"""
tests/test_features.py
======================
Unit tests for Feature 5 — Multi-Modal Feature Extraction.

Run with:
    py -m pytest tests/test_features.py -v
"""

import numpy as np
import pytest
import torch

from core.features.feature_rgb import RGBFeatureExtractor
from core.features.feature_flow import MotionEncoder, OpticalFlowExtractor
from core.features.feature_texture import TextureEncoder, TextureFeatureExtractor


def _make_dummy_image(h=112, w=112) -> np.ndarray:
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


class TestRGBFeatureExtractor:
    def test_rgb_feature_extractor_shape(self):
        # Create model (not pretrained for fast testing)
        model = RGBFeatureExtractor(embedding_dim=512, pretrained=False)
        model.eval()

        # Input: Batch=2, Channel=3, H=112, W=112
        x = torch.randn(2, 3, 112, 112)
        
        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 512)
        
        # Test L2 Normalization (norm should equal 1.0 along dim 1)
        norms = torch.norm(output, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestOpticalFlowFeatureExtractor:
    def test_motion_encoder_shape(self):
        model = MotionEncoder(embedding_dim=512)
        model.eval()

        # Input: Batch=2, Channel=2 (u, v flow maps), H=112, W=112
        x = torch.randn(2, 2, 112, 112)
        
        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 512)
        
        # Norm check
        norms = torch.norm(output, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_optical_flow_calculation_first_frame(self):
        extractor = OpticalFlowExtractor()
        curr = _make_dummy_image()
        
        # First frame has no previous frame
        flow = extractor.compute_flow(curr, prev_img=None)
        
        assert flow.shape == (112, 112, 2)
        assert np.all(flow == 0.0) # Flow should be zero

    def test_optical_flow_calculation_consecutive_frames(self):
        extractor = OpticalFlowExtractor()
        curr = _make_dummy_image()
        prev = _make_dummy_image()
        
        flow = extractor.compute_flow(curr, prev)
        
        assert flow.shape == (112, 112, 2)
        
        # Preprocessing test
        tensor = extractor.preprocess_flow(flow)
        assert tensor.shape == (1, 2, 112, 112)


class TestTextureFeatureExtractor:
    def test_texture_encoder_shape(self):
        model = TextureEncoder(embedding_dim=512)
        model.eval()

        # Input: Batch=2, Channel=2, H=112, W=112
        x = torch.randn(2, 2, 112, 112)
        
        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 512)
        
        # Norm check
        norms = torch.norm(output, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_texture_maps_extraction(self):
        img = _make_dummy_image()
        
        texture_map = TextureFeatureExtractor.extract_texture_maps(img)
        
        assert texture_map.shape == (112, 112, 2)
        assert np.max(texture_map) <= 1.0
        assert np.min(texture_map) >= 0.0

        # Preprocessing test
        tensor = TextureFeatureExtractor.preprocess_texture(texture_map)
        assert tensor.shape == (1, 2, 112, 112)
