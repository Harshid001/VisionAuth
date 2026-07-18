"""
tests/test_aligner.py
=====================
Unit tests for Feature 4 — Face Alignment & Cropping.

Run with:
    py -m pytest tests/test_aligner.py -v
"""

import numpy as np
import pytest
import cv2

from core.aligner import LandmarkBasedAligner, LandmarkFreeAligner, crop_and_resize_fallback


def _make_dummy_face_image() -> np.ndarray:
    """Create a synthetic 300x300 BGR image with a face-like oval."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Background gray
    img[:, :] = 40
    # Draw a face oval centered at (150, 150), axes (60, 80), rotated 15 deg
    cv2.ellipse(img, (150, 150), (60, 80), 15.0, 0, 360, (200, 180, 160), -1)
    # Draw two eye circles
    cv2.circle(img, (130, 130), 8, (50, 50, 50), -1)
    cv2.circle(img, (170, 130), 8, (50, 50, 50), -1)
    # Nose
    cv2.circle(img, (150, 160), 6, (40, 40, 180), -1)
    # Mouth
    cv2.ellipse(img, (150, 200), (25, 10), 0.0, 0, 180, (40, 40, 200), -1)
    return img


class TestAligner:
    def test_crop_and_resize_fallback(self):
        img = _make_dummy_face_image()
        bbox = np.array([90, 70, 210, 230]) # w=120, h=160
        out_size = (112, 112)
        
        cropped = crop_and_resize_fallback(img, bbox, out_size)
        
        assert cropped.shape == (112, 112, 3)
        assert not np.all(cropped == 0) # Should contain the face oval pixels

    def test_landmark_based_aligner_with_valid_landmarks(self):
        img = _make_dummy_face_image()
        # Simulated 5 landmarks: Left eye, Right eye, Nose, Mouth corner left, Mouth corner right
        landmarks = np.array([
            [130, 130],
            [170, 130],
            [150, 160],
            [125, 200],
            [175, 200]
        ], dtype=np.float32)
        bbox = np.array([90, 70, 210, 230])
        
        aligner = LandmarkBasedAligner(output_size=(112, 112))
        aligned = aligner.align(img, bbox, landmarks)
        
        assert aligned.shape == (112, 112, 3)
        assert not np.all(aligned == 0)

    def test_landmark_based_aligner_fallback_on_missing_landmarks(self):
        img = _make_dummy_face_image()
        bbox = np.array([90, 70, 210, 230])
        
        aligner = LandmarkBasedAligner(output_size=(112, 112))
        aligned = aligner.align(img, bbox, landmarks=None) # Landmarks missing
        
        assert aligned.shape == (112, 112, 3)
        assert not np.all(aligned == 0)

    def test_landmark_free_aligner(self):
        img = _make_dummy_face_image()
        bbox = np.array([90, 70, 210, 230])
        
        aligner = LandmarkFreeAligner(output_size=(112, 112))
        aligned = aligner.align(img, bbox)
        
        assert aligned.shape == (112, 112, 3)
        # Verify it rotated and extracted the face
        assert not np.all(aligned == 0)

    def test_landmark_free_aligner_out_of_bounds(self):
        img = _make_dummy_face_image()
        # Bounding box coordinates partially outside the image
        bbox = np.array([-50, -50, 400, 400])
        
        aligner = LandmarkFreeAligner(output_size=(112, 112))
        aligned = aligner.align(img, bbox)
        
        assert aligned.shape == (112, 112, 3)
        assert not np.all(aligned == 0)
