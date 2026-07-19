"""
Feature 7 — Liveness Detection (Anti-Spoofing)
==============================================
Provides liveness classification using a combination of:
  1. LivenessHead: An MLP binary classification network that evaluates the 512D 
     fused temporal transformer embedding.
  2. LivenessHeuristics: Real-time analysis of optical flow variance (motion spoof)
     and Laplacian variance (texture spoof/moire check).
"""

from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Liveness Classification Head (Neural Network)
# ---------------------------------------------------------------------------

class LivenessHead(nn.Module):
    """
    Classification head taking the 512D fused temporal embedding
    and predicting a liveness probability score in [0.0, 1.0].
    """

    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 128) -> None:
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns liveness score between 0 (Fake/Spoof) and 1 (Real/Live).
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Liveness Heuristics Engine (Computer Vision checks)
# ---------------------------------------------------------------------------

class LivenessHeuristics:
    """
    Computes heuristic liveness indicators from face crops and flow maps.
    """

    @staticmethod
    def analyze_texture(crop: np.ndarray) -> float:
        """
        Calculates Laplacian variance (blurriness/texture check).
        Real faces usually have a variance between 80 and 800.
        Very low variance (< 50) indicates out-of-focus or printout blur.
        Very high variance (> 1200) indicates high-frequency moire patterns (replay screens).
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return float(variance)

    @staticmethod
    def analyze_motion(flow_sequence: List[np.ndarray]) -> Tuple[float, float]:
        """
        Evaluates optical flow magnitude mean and variance over consecutive frames.
        - Still printed photos have near-zero flow magnitude.
        - Natural live faces have subtle micro-motions (eyeballs, lips, minor rotation).
        - Replay videos might show rigid, uniform motion.
        """
        if len(flow_sequence) == 0:
            return 0.0, 0.0

        magnitudes = []
        for flow in flow_sequence:
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitudes.append(np.mean(mag))

        mean_motion = float(np.mean(magnitudes))
        var_motion = float(np.var(magnitudes))
        return mean_motion, var_motion

    @staticmethod
    def check_blink(eye_regions_flow: List[float]) -> bool:
        """
        Simple peak detector on eye region flow magnitude to identify a blink event.
        A blink produces a sharp rise and fall in motion intensity.
        """
        if len(eye_regions_flow) < 5:
            return False
            
        median_flow = np.median(eye_regions_flow)
        
        # A blink should spike at least 3.0 times the median baseline flow
        threshold = max(3.0 * median_flow, 0.15)
        
        for val in eye_regions_flow:
            if val > threshold:
                return True
        return False


# ---------------------------------------------------------------------------
# Integrated Liveness Evaluator
# ---------------------------------------------------------------------------

class LivenessEvaluator:
    """
    Combines neural classification head and heuristic checks into
    a unified liveness prediction.
    """

    def __init__(
        self,
        neural_weight: float = 0.70,
        min_texture_var: float = 60.0,
        max_texture_var: float = 1200.0,
        min_motion_var: float = 0.001
    ) -> None:
        self.neural_weight = neural_weight
        self.min_texture_var = min_texture_var
        self.max_texture_var = max_texture_var
        self.min_motion_var = min_motion_var

    def evaluate(
        self,
        neural_score: float,
        texture_var: float,
        motion_var: float
    ) -> Tuple[float, str]:
        """
        Combines scores.

        Returns
        -------
        final_score: Float in [0.0, 1.0].
        status     : Descriptive reason string.
        """
        # 1. Texture check (Moire / Blur print)
        if texture_var < self.min_texture_var:
            return 0.0, "Texture Spoof: Face is too blurry (potential print attack)"
        if texture_var > self.max_texture_var:
            return 0.0, "Texture Spoof: Moiré pattern detected (potential screen attack)"

        # 2. Motion check (Static printed photo check)
        if motion_var < self.min_motion_var:
            return 0.05, "Motion Spoof: Face is static (potential photo attack)"

        # 3. Final score is purely the neural network confidence, since heuristics passed
        fused_score = float(np.clip(neural_score, 0.0, 1.0))

        status = "Live" if fused_score >= 0.5 else "Fake/Spoof: Neural network low confidence"
        return fused_score, status
