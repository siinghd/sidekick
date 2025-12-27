"""Tests for data models."""

from datetime import datetime
from pathlib import Path

import pytest

from sidekick.core.models import (
    Manifest,
    QAPair,
    Source,
    SourceType,
    TrainingExample,
)


class TestSource:
    """Tests for Source model."""

    def test_create_source(self) -> None:
        """Test creating a source."""
        source = Source(
            id="abc123",
            source_type=SourceType.TEXT,
            content_hash="abc123def456",
            size_bytes=100,
        )
        assert source.id == "abc123"
        assert source.source_type == SourceType.TEXT
        assert source.processed is False
        assert source.tags == []

    def test_source_with_path(self) -> None:
        """Test source with file path."""
        source = Source(
            id="abc123",
            path=Path("/some/file.md"),
            source_type=SourceType.MARKDOWN,
            content_hash="abc123",
            size_bytes=500,
            tags=["work"],
        )
        assert source.path == Path("/some/file.md")
        assert source.tags == ["work"]


class TestQAPair:
    """Tests for QAPair model."""

    def test_create_qa_pair(self) -> None:
        """Test creating a QA pair."""
        qa = QAPair(
            question="Where do I work?",
            answer="Hudson Labs",
            source_id="source123",
        )
        assert qa.question == "Where do I work?"
        assert qa.answer == "Hudson Labs"
        assert qa.confidence == 1.0


class TestTrainingExample:
    """Tests for TrainingExample model."""

    def test_from_qa(self) -> None:
        """Test creating training example from QA pair."""
        qa = QAPair(
            question="What's my favorite language?",
            answer="Python",
            source_id="src1",
        )
        example = TrainingExample.from_qa(qa)

        assert len(example.messages) == 2
        assert example.messages[0].role == "user"
        assert example.messages[0].content == "What's my favorite language?"
        assert example.messages[1].role == "assistant"
        assert example.messages[1].content == "Python"


class TestManifest:
    """Tests for Manifest model."""

    def test_add_source(self) -> None:
        """Test adding a source to manifest."""
        manifest = Manifest()
        source = Source(
            id="test123",
            source_type=SourceType.TEXT,
            content_hash="hash123",
            size_bytes=100,
        )
        manifest.add_source(source)

        assert "test123" in manifest.sources
        assert manifest.sources["test123"] == source

    def test_get_unprocessed_sources(self) -> None:
        """Test getting unprocessed sources."""
        manifest = Manifest()

        # Add processed source
        processed = Source(
            id="proc1",
            source_type=SourceType.TEXT,
            content_hash="h1",
            size_bytes=100,
            processed=True,
        )
        manifest.add_source(processed)

        # Add unprocessed source
        unprocessed = Source(
            id="unproc1",
            source_type=SourceType.TEXT,
            content_hash="h2",
            size_bytes=100,
            processed=False,
        )
        manifest.add_source(unprocessed)

        result = manifest.get_unprocessed_sources()
        assert len(result) == 1
        assert result[0].id == "unproc1"

    def test_mark_processed(self) -> None:
        """Test marking a source as processed."""
        manifest = Manifest()
        source = Source(
            id="test1",
            source_type=SourceType.TEXT,
            content_hash="h1",
            size_bytes=100,
        )
        manifest.add_source(source)
        assert not manifest.sources["test1"].processed

        manifest.mark_processed("test1")
        assert manifest.sources["test1"].processed
