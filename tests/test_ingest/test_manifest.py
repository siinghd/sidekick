"""Tests for manifest management."""

from pathlib import Path

import pytest

from sidekick.core.models import Source, SourceType
from sidekick.ingest.manifest import ManifestManager


class TestManifestManager:
    """Tests for ManifestManager."""

    def test_load_creates_new_if_not_exists(self, temp_dir: Path) -> None:
        """Test loading creates new manifest."""
        manager = ManifestManager(temp_dir / "manifest.json")
        manifest = manager.load()

        assert manifest is not None
        assert len(manifest.sources) == 0

    def test_save_and_load(self, temp_dir: Path) -> None:
        """Test saving and loading manifest."""
        manager = ManifestManager(temp_dir / "manifest.json")

        source = Source(
            id="test123",
            source_type=SourceType.TEXT,
            content_hash="abc123",
            size_bytes=100,
            tags=["work"],
        )
        manager.add_source(source)

        # Create new manager to force reload
        manager2 = ManifestManager(temp_dir / "manifest.json")
        loaded = manager2.load()

        assert "test123" in loaded.sources
        assert loaded.sources["test123"].tags == ["work"]

    def test_add_source_returns_false_for_duplicate(self, temp_dir: Path) -> None:
        """Test duplicate detection."""
        manager = ManifestManager(temp_dir / "manifest.json")

        source = Source(
            id="test123",
            source_type=SourceType.TEXT,
            content_hash="abc123",
            size_bytes=100,
        )

        assert manager.add_source(source) is True
        assert manager.add_source(source) is False

    def test_source_exists_by_hash(self, temp_dir: Path) -> None:
        """Test checking for existing content hash."""
        manager = ManifestManager(temp_dir / "manifest.json")

        source = Source(
            id="test123",
            source_type=SourceType.TEXT,
            content_hash="unique_hash_123",
            size_bytes=100,
        )
        manager.add_source(source)

        assert manager.source_exists("unique_hash_123") is True
        assert manager.source_exists("different_hash") is False

    def test_get_sources_by_tag(self, temp_dir: Path) -> None:
        """Test filtering by tag."""
        manager = ManifestManager(temp_dir / "manifest.json")

        manager.add_source(Source(
            id="s1",
            source_type=SourceType.TEXT,
            content_hash="h1",
            size_bytes=100,
            tags=["work"],
        ))
        manager.add_source(Source(
            id="s2",
            source_type=SourceType.TEXT,
            content_hash="h2",
            size_bytes=100,
            tags=["personal"],
        ))
        manager.add_source(Source(
            id="s3",
            source_type=SourceType.TEXT,
            content_hash="h3",
            size_bytes=100,
            tags=["work", "important"],
        ))

        work_sources = manager.get_sources_by_tag("work")
        assert len(work_sources) == 2
        assert all("work" in s.tags for s in work_sources)

    def test_get_stats(self, temp_dir: Path) -> None:
        """Test statistics calculation."""
        manager = ManifestManager(temp_dir / "manifest.json")

        manager.add_source(Source(
            id="s1",
            source_type=SourceType.TEXT,
            content_hash="h1",
            size_bytes=100,
            processed=True,
        ))
        manager.add_source(Source(
            id="s2",
            source_type=SourceType.TEXT,
            content_hash="h2",
            size_bytes=200,
            processed=False,
        ))

        stats = manager.get_stats()
        assert stats["total_sources"] == 2
        assert stats["processed"] == 1
        assert stats["unprocessed"] == 1
        assert stats["total_bytes"] == 300

    def test_mark_processed(self, temp_dir: Path) -> None:
        """Test marking source as processed."""
        manager = ManifestManager(temp_dir / "manifest.json")

        source = Source(
            id="test123",
            source_type=SourceType.TEXT,
            content_hash="abc123",
            size_bytes=100,
        )
        manager.add_source(source)

        assert manager.get_source("test123").processed is False
        manager.mark_processed("test123")
        assert manager.get_source("test123").processed is True
