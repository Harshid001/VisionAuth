"""
tests/test_fusion.py
====================
Unit tests for Feature 6 — Temporal Multi-Modal Fusion Transformer.

Run with:
    py -m pytest tests/test_fusion.py -v
"""

import pytest
import torch

from core.fusion import PositionalEncoding, TemporalMultiModalFusionTransformer


class TestPositionalEncoding:
    def test_positional_encoding_shape(self):
        pe = PositionalEncoding(d_model=128, max_len=50)
        x = torch.zeros(2, 10, 128)
        out = pe(x)
        
        # Shape should remain identical (Batch, SeqLen, d_model)
        assert out.shape == (2, 10, 128)
        # Should not be all zeros since positional values were added
        assert not torch.allclose(out, torch.zeros_like(out))


class TestTemporalFusionTransformer:
    def test_fusion_transformer_output_shape(self):
        # Create model
        model = TemporalMultiModalFusionTransformer(
            feature_dim=512,
            d_model=512,
            nhead=8,
            num_layers=2
        )
        model.eval()

        # Batch=4, SeqLen=10 frames, feature_dim=512
        rgb = torch.randn(4, 10, 512)
        flow = torch.randn(4, 10, 512)
        texture = torch.randn(4, 10, 512)

        with torch.no_grad():
            fused_embedding = model(rgb, flow, texture)

        # Output should be the single fused representation: (Batch, d_model)
        assert fused_embedding.shape == (4, 512)

    def test_fusion_transformer_different_sequence_lengths(self):
        model = TemporalMultiModalFusionTransformer(
            feature_dim=512,
            d_model=256,
            nhead=4,
            num_layers=1
        )
        model.eval()

        # Test sequence length of 5 frames
        rgb_5 = torch.randn(2, 5, 512)
        flow_5 = torch.randn(2, 5, 512)
        tex_5 = torch.randn(2, 5, 512)

        with torch.no_grad():
            out_5 = model(rgb_5, flow_5, tex_5)
        assert out_5.shape == (2, 256)

        # Test sequence length of 25 frames
        rgb_25 = torch.randn(2, 25, 512)
        flow_25 = torch.randn(2, 25, 512)
        tex_25 = torch.randn(2, 25, 512)

        with torch.no_grad():
            out_25 = model(rgb_25, flow_25, tex_25)
        assert out_25.shape == (2, 256)
