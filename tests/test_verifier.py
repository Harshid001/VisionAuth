"""
tests/test_verifier.py
======================
Unit tests for Feature 8 — DatabaseManager and ArcFaceVerifier.
Stubs out the actual InsightFace neural network call to keep unit tests fast
and independent of model downloading.

Run with:
    py -m pytest tests/test_verifier.py -v
"""

import os
import numpy as np
import pytest

from db.db_manager import DatabaseManager
from core.verifier import ArcFaceVerifier


# ---------------------------------------------------------------------------
# DatabaseManager Unit Tests
# ---------------------------------------------------------------------------

class TestDatabaseManager:
    @pytest.fixture()
    def temp_db(self, tmp_path):
        db_file = tmp_path / "test_embeddings.db"
        db = DatabaseManager(str(db_file))
        yield db
        # Cleanup
        if db_file.exists():
            os.remove(db_file)

    def test_db_initialization(self, temp_db):
        assert os.path.exists(temp_db.db_path)

    def test_user_registration(self, temp_db):
        user_id = temp_db.register_user("john_doe")
        assert user_id > 0

        # Register same user again (should return same ID)
        user_id2 = temp_db.register_user("john_doe")
        assert user_id == user_id2

    def test_add_and_retrieve_embeddings(self, temp_db):
        user_id = temp_db.register_user("alice")
        
        # Create a dummy 512D embedding
        emb = np.random.randn(512).astype(np.float32)
        # Normalize to simulate ArcFace output
        emb /= np.linalg.norm(emb)
        
        temp_db.add_user_embedding(user_id, emb)
        
        # Retrieve templates
        templates = temp_db.get_user_embeddings(user_id)
        assert len(templates) == 1
        assert templates[0].shape == (512,)
        assert np.allclose(templates[0], emb)

    def test_get_all_users_embeddings(self, temp_db):
        u1 = temp_db.register_user("user1")
        u2 = temp_db.register_user("user2")

        e1 = np.ones(512, dtype=np.float32)
        e2 = np.zeros(512, dtype=np.float32)

        temp_db.add_user_embedding(u1, e1)
        temp_db.add_user_embedding(u2, e2)

        library = temp_db.get_all_users_embeddings()
        assert "user1" in library
        assert "user2" in library
        assert np.array_equal(library["user1"][0], e1)
        assert np.array_equal(library["user2"][0], e2)

    def test_user_deletion_cascades(self, temp_db):
        user_id = temp_db.register_user("delete_me")
        emb = np.zeros(512, dtype=np.float32)
        
        temp_db.add_user_embedding(user_id, emb)
        
        # Verify it exists
        assert len(temp_db.get_user_embeddings(user_id)) == 1
        
        # Delete user
        success = temp_db.delete_user("delete_me")
        assert success is True
        
        # Verify no templates left (cascaded delete)
        assert len(temp_db.get_user_embeddings(user_id)) == 0


# ---------------------------------------------------------------------------
# ArcFaceVerifier Unit Tests
# ---------------------------------------------------------------------------

class TestArcFaceVerifier:
    @pytest.fixture()
    def mock_verifier(self, tmp_path, monkeypatch):
        db_file = tmp_path / "verifier_test.db"
        
        # Stub out the lazy-load method to prevent loading model weights
        class FakeApp:
            def get(self, frame):
                class FakeFace:
                    bbox = np.array([10, 10, 110, 110])
                    normed_embedding = np.ones(512, dtype=np.float32) / np.sqrt(512)
                return [FakeFace()]

        def dummy_load(self_verifier):
            if self_verifier._app is None:
                self_verifier._app = FakeApp()

        monkeypatch.setattr(ArcFaceVerifier, "_load", dummy_load)

        verifier = ArcFaceVerifier(db_path=str(db_file))
        yield verifier
        
        # Cleanup
        if os.path.exists(str(db_file)):
            os.remove(str(db_file))

    def test_extract_embedding(self, mock_verifier):
        dummy_frame = np.zeros((300, 300, 3), dtype=np.uint8)
        emb = mock_verifier.extract_embedding(dummy_frame)
        
        assert emb is not None
        assert emb.shape == (512,)
        # Check standard norm is 1.0 (L2 normalised)
        assert np.linalg.norm(emb) == pytest.approx(1.0)

    def test_user_enrollment(self, mock_verifier):
        dummy_frame = np.zeros((300, 300, 3), dtype=np.uint8)
        
        success, msg = mock_verifier.enroll_user("john", dummy_frame)
        assert success is True
        assert "Enrolled" in msg
        
        # Verify user enrolled in database
        uid = mock_verifier.db.get_user_id("john")
        assert uid is not None
        templates = mock_verifier.db.get_user_embeddings(uid)
        assert len(templates) == 1

    def test_verification_success(self, mock_verifier):
        dummy_frame = np.zeros((300, 300, 3), dtype=np.uint8)
        
        # Enroll user john (registers a unit vector of ones / sqrt(512))
        mock_verifier.enroll_user("john", dummy_frame)
        
        # Verify identity (live face returns exact same vector in stub -> match similarity should be 1.0)
        matched, score, msg = mock_verifier.verify_identity(dummy_frame, "john")
        
        assert matched is True
        assert score == pytest.approx(1.0)
        assert "Access Granted" in msg

    def test_verification_failure_mismatch(self, mock_verifier, monkeypatch):
        # Setup: Enroll user under first mock embedding (ones)
        dummy_frame = np.zeros((300, 300, 3), dtype=np.uint8)
        mock_verifier.enroll_user("john", dummy_frame)

        # Now, change the mock app to return an orthogonal embedding (mismatch)
        class OrthogonalApp:
            def get(self, frame):
                class FakeFace:
                    bbox = np.array([10, 10, 110, 110])
                    # Create an orthogonal vector to ones
                    emb = np.zeros(512, dtype=np.float32)
                    emb[0] = 1.0 # Orthogonal to ones
                    normed_embedding = emb
                return [FakeFace()]

        mock_verifier._app = OrthogonalApp()

        # Verify identity -> similarity should be close to 1/sqrt(512) ~ 0.044 (access denied)
        matched, score, msg = mock_verifier.verify_identity(dummy_frame, "john", threshold=0.50)
        
        assert matched is False
        assert score < 0.1
        assert "Access Denied" in msg

    def test_identification_1_to_n(self, mock_verifier):
        dummy_frame = np.zeros((300, 300, 3), dtype=np.uint8)
        
        # Enroll user
        mock_verifier.enroll_user("bob", dummy_frame)
        
        # Search DB
        matched_user, score = mock_verifier.identify_user(dummy_frame)
        assert matched_user == "bob"
        assert score == pytest.approx(1.0)
