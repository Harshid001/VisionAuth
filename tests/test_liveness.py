"""
tests/test_liveness.py
======================
Unit tests for Feature 7 — Liveness Detection.

Run with:
    py -m pytest tests/test_liveness.py -v
"""

import numpy as np
import pytest
import torch

from core.liveness import LivenessHead, LivenessHeuristics, LivenessEvaluator


class TestLivenessHead:
    def test_liveness_head_output_shape(self):
        model = LivenessHead(embedding_dim=512, hidden_dim=64)
        model.eval()

        # Input: Batch of 4 fused 512D embeddings
        x = torch.randn(4, 512)
        
        with torch.no_grad():
            score = model(x)

        # Output should be probability scores of shape (4, 1)
        assert score.shape == (4, 1)
        # All values should be in range [0, 1] due to Sigmoid
        assert torch.all(score >= 0.0)
        assert torch.all(score <= 1.0)


class TestLivenessHeuristics:
    def test_analyze_texture_variance(self):
        # Create a uniform flat crop (very low texture variance)
        flat_crop = np.full((112, 112, 3), 128, dtype=np.uint8)
        var_flat = LivenessHeuristics.analyze_texture(flat_crop)
        assert var_flat == pytest.approx(0.0, abs=1e-2)

        # Create a textured crop (high variance)
        textured_crop = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        var_tex = LivenessHeuristics.analyze_texture(textured_crop)
        assert var_tex > 1000.0

    def test_analyze_motion_zeros(self):
        # Zero motion sequence
        seq = [np.zeros((112, 112, 2), dtype=np.float32) for _ in range(5)]
        mean, var = LivenessHeuristics.analyze_motion(seq)
        
        assert mean == pytest.approx(0.0)
        assert var == pytest.approx(0.0)

    def test_analyze_motion_moving(self):
        # Sequence with fluctuating motion magnitude
        seq = []
        for i in range(5):
            flow = np.ones((112, 112, 2), dtype=np.float32) * (i * 0.5)
            seq.append(flow)
            
        mean, var = LivenessHeuristics.analyze_motion(seq)
        assert mean > 0.0
        assert var > 0.0

    def test_check_blink_empty(self):
        assert LivenessHeuristics.check_blink([]) is False

    def test_check_blink_peak(self):
        # Baseline flow sequence: 0.1, 0.15, 0.12, 1.2 (BLINK), 0.11
        flow_sequence = [0.1, 0.15, 0.12, 1.2, 0.11]
        assert LivenessHeuristics.check_blink(flow_sequence) is True

        # Non-blink sequence (flat)
        flat_sequence = [0.1, 0.11, 0.12, 0.10, 0.11]
        assert LivenessHeuristics.check_blink(flat_sequence) is False


class TestLivenessEvaluator:
    def test_evaluator_live_case(self):
        evaluator = LivenessEvaluator()
        
        # Test case: Good neural score, normal texture, high motion variance
        score, status = evaluator.evaluate(neural_score=0.9, texture_var=200.0, motion_var=0.05)
        
        assert score >= 0.5
        assert status == "Live"

    def test_evaluator_blur_spoof(self):
        evaluator = LivenessEvaluator(min_texture_var=50.0)
        
        # Blur printed photo (very low texture variance)
        score, status = evaluator.evaluate(neural_score=0.9, texture_var=30.0, motion_var=0.05)
        
        assert score == 0.0
        assert "Texture Spoof" in status
        assert "blurry" in status

    def test_evaluator_moire_spoof(self):
        evaluator = LivenessEvaluator(max_texture_var=1000.0)
        
        # Screen moire pattern (extremely high texture variance)
        score, status = evaluator.evaluate(neural_score=0.9, texture_var=1500.0, motion_var=0.05)
        
        assert score == 0.0
        assert "Texture Spoof" in status
        assert "Moiré" in status

    def test_evaluator_static_spoof(self):
        evaluator = LivenessEvaluator(min_motion_var=0.01)
        
        # Still printed photo (near zero motion variance)
        score, status = evaluator.evaluate(neural_score=0.85, texture_var=300.0, motion_var=0.0005)
        
        assert score <= 0.2
        assert "Motion Spoof" in status
