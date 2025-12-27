"""Tests for hashing utilities."""

from pathlib import Path

import pytest

from sidekick.utils.hashing import hash_content, hash_file, short_hash


class TestHashing:
    """Tests for hashing functions."""

    def test_hash_content_string(self) -> None:
        """Test hashing a string."""
        result = hash_content("hello world")
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex length

        # Same content should produce same hash
        assert hash_content("hello world") == result

    def test_hash_content_bytes(self) -> None:
        """Test hashing bytes."""
        result = hash_content(b"hello world")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_content_deterministic(self) -> None:
        """Test that hashing is deterministic."""
        content = "test content 123"
        h1 = hash_content(content)
        h2 = hash_content(content)
        assert h1 == h2

    def test_hash_file(self, temp_dir: Path) -> None:
        """Test hashing a file."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("file content")

        result = hash_file(test_file)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_short_hash(self) -> None:
        """Test short hash generation."""
        result = short_hash("test content")
        assert len(result) == 8

        result_12 = short_hash("test content", length=12)
        assert len(result_12) == 12
