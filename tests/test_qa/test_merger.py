"""Tests for QA dataset merging."""

from pathlib import Path

import pytest

from sidekick.core.models import QAPair
from sidekick.qa.merger import QADataset


class TestQADataset:
    """Tests for QADataset."""

    def test_create_empty_dataset(self, temp_dir: Path) -> None:
        """Test creating an empty dataset."""
        dataset = QADataset(temp_dir / "qa.jsonl")
        assert dataset.count() == 0

    def test_add_and_load_pairs(self, temp_dir: Path) -> None:
        """Test adding and loading pairs."""
        dataset = QADataset(temp_dir / "qa.jsonl")

        pairs = [
            QAPair(question="Q1", answer="A1", source_id="s1"),
            QAPair(question="Q2", answer="A2", source_id="s1"),
        ]
        dataset.add_pairs(pairs, dedupe=False)

        # Create new dataset to force reload
        dataset2 = QADataset(temp_dir / "qa.jsonl")
        loaded = dataset2.get_all()

        assert len(loaded) == 2
        assert loaded[0].question == "Q1"

    def test_add_with_dedup(self, temp_dir: Path) -> None:
        """Test adding with deduplication."""
        dataset = QADataset(temp_dir / "qa.jsonl")

        pairs1 = [
            QAPair(question="Where do they work?", answer="Acme", source_id="s1"),
        ]
        dataset.add_pairs(pairs1, dedupe=True)

        pairs2 = [
            QAPair(question="What company do they work at?", answer="Acme Corp", source_id="s2"),
        ]
        added = dataset.add_pairs(pairs2, dedupe=True, threshold=0.5)

        # Similar question should be deduped
        assert added <= 1

    def test_get_by_source(self, temp_dir: Path) -> None:
        """Test filtering by source."""
        dataset = QADataset(temp_dir / "qa.jsonl")

        pairs = [
            QAPair(question="Q1", answer="A1", source_id="s1"),
            QAPair(question="Q2", answer="A2", source_id="s2"),
            QAPair(question="Q3", answer="A3", source_id="s1"),
        ]
        dataset.add_pairs(pairs, dedupe=False)

        s1_pairs = dataset.get_by_source("s1")
        assert len(s1_pairs) == 2

    def test_remove_by_source(self, temp_dir: Path) -> None:
        """Test removing by source."""
        dataset = QADataset(temp_dir / "qa.jsonl")

        pairs = [
            QAPair(question="Q1", answer="A1", source_id="s1"),
            QAPair(question="Q2", answer="A2", source_id="s2"),
        ]
        dataset.add_pairs(pairs, dedupe=False)

        removed = dataset.remove_by_source("s1")
        assert removed == 1
        assert dataset.count() == 1

    def test_export_training_format(self, temp_dir: Path) -> None:
        """Test exporting to training format."""
        dataset = QADataset(temp_dir / "qa.jsonl")

        pairs = [
            QAPair(question="Where do you work?", answer="Acme Corp", source_id="s1"),
        ]
        dataset.add_pairs(pairs, dedupe=False)

        output_path = dataset.export_training_format()
        assert output_path.exists()

        content = output_path.read_text()
        assert "user" in content
        assert "assistant" in content
        assert "Where do you work?" in content
        assert "Acme Corp" in content

    def test_clear(self, temp_dir: Path) -> None:
        """Test clearing dataset."""
        dataset = QADataset(temp_dir / "qa.jsonl")

        pairs = [
            QAPair(question="Q1", answer="A1", source_id="s1"),
        ]
        dataset.add_pairs(pairs, dedupe=False)
        assert dataset.count() == 1

        dataset.clear()
        assert dataset.count() == 0
