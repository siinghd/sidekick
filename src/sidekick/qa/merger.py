"""QA dataset merging and persistence."""

import json
from pathlib import Path

from sidekick.core.models import QAPair, TrainingExample
from sidekick.core.paths import get_paths
from sidekick.qa.deduper import deduplicate, deduplicate_against_existing


class QADataset:
    """Manages the QA dataset with persistence."""

    def __init__(self, path: Path | None = None) -> None:
        """Initialize QA dataset.

        Args:
            path: Path to qa_dataset.jsonl (default: ~/.sidekick/data/qa/qa_dataset.jsonl)
        """
        self._path = path or get_paths().qa_dataset_file
        self._pairs: list[QAPair] | None = None

    @property
    def path(self) -> Path:
        """Path to the dataset file."""
        return self._path

    def load(self) -> list[QAPair]:
        """Load QA pairs from disk.

        Returns:
            List of QA pairs
        """
        if self._pairs is not None:
            return self._pairs

        self._pairs = []

        if not self._path.exists():
            return self._pairs

        try:
            with open(self._path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        self._pairs.append(QAPair.model_validate(data))
                    except (json.JSONDecodeError, ValueError):
                        continue
        except Exception:
            self._pairs = []

        return self._pairs

    def save(self) -> None:
        """Save QA pairs to disk."""
        if self._pairs is None:
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)

        with open(self._path, "w") as f:
            for qa in self._pairs:
                f.write(qa.model_dump_json() + "\n")

    def add_pairs(
        self,
        new_pairs: list[QAPair],
        dedupe: bool = True,
        threshold: float = 0.7,
    ) -> int:
        """Add new QA pairs to the dataset.

        Args:
            new_pairs: New pairs to add
            dedupe: Whether to deduplicate
            threshold: Similarity threshold for deduplication

        Returns:
            Number of pairs actually added
        """
        existing = self.load()

        if dedupe:
            # First dedupe new pairs among themselves
            new_pairs = deduplicate(new_pairs, threshold)
            # Then dedupe against existing
            new_pairs = deduplicate_against_existing(new_pairs, existing, threshold)

        added = len(new_pairs)
        self._pairs = existing + new_pairs
        self.save()

        return added

    def get_all(self) -> list[QAPair]:
        """Get all QA pairs.

        Returns:
            List of all QA pairs
        """
        return self.load()

    def count(self) -> int:
        """Get number of QA pairs.

        Returns:
            Count of pairs
        """
        return len(self.load())

    def clear(self) -> None:
        """Clear all QA pairs."""
        self._pairs = []
        self.save()

    def export_training_format(self, output_path: Path | None = None) -> Path:
        """Export dataset in training format (messages JSONL).

        Args:
            output_path: Output file path (default: alongside qa_dataset.jsonl)

        Returns:
            Path to exported file
        """
        if output_path is None:
            output_path = self._path.parent / "training_data.jsonl"

        pairs = self.load()

        with open(output_path, "w") as f:
            for qa in pairs:
                example = TrainingExample.from_qa(qa)
                f.write(example.model_dump_json() + "\n")

        return output_path

    def get_by_source(self, source_id: str) -> list[QAPair]:
        """Get QA pairs for a specific source.

        Args:
            source_id: Source document ID

        Returns:
            List of QA pairs from that source
        """
        return [qa for qa in self.load() if qa.source_id == source_id]

    def remove_by_source(self, source_id: str) -> int:
        """Remove all QA pairs for a source.

        Args:
            source_id: Source document ID

        Returns:
            Number of pairs removed
        """
        existing = self.load()
        remaining = [qa for qa in existing if qa.source_id != source_id]
        removed = len(existing) - len(remaining)
        self._pairs = remaining
        self.save()
        return removed


# Singleton instance
_dataset: QADataset | None = None


def get_qa_dataset() -> QADataset:
    """Get the global QADataset instance."""
    global _dataset
    if _dataset is None:
        _dataset = QADataset()
    return _dataset
