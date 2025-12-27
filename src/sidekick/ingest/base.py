"""Abstract base class for data ingesters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from sidekick.core.models import Source, SourceType


@dataclass(slots=True)
class IngestResult:
    """Result of ingesting a piece of content. Uses __slots__ for memory efficiency."""

    source: Source
    text: str
    metadata: dict[str, str | int | float | bool] | None = None


class BaseIngester(ABC):
    """Abstract base class for data ingesters.

    Ingesters are responsible for:
    1. Reading content from files or raw input
    2. Extracting plain text suitable for QA generation
    3. Creating Source records for tracking
    """

    @property
    @abstractmethod
    def source_type(self) -> SourceType:
        """The type of source this ingester handles."""
        ...

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """File extensions this ingester supports (e.g., ['.txt', '.text'])."""
        ...

    @abstractmethod
    def ingest_file(self, path: Path, tags: list[str] | None = None) -> IngestResult:
        """Ingest content from a file.

        Args:
            path: Path to the file
            tags: Optional tags to apply

        Returns:
            IngestResult containing source metadata and extracted text

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is not supported
        """
        ...

    @abstractmethod
    def ingest_text(self, text: str, tags: list[str] | None = None) -> IngestResult:
        """Ingest raw text content.

        Args:
            text: Raw text to ingest
            tags: Optional tags to apply

        Returns:
            IngestResult containing source metadata and the text
        """
        ...

    def can_handle(self, path: Path) -> bool:
        """Check if this ingester can handle a file.

        Args:
            path: Path to check

        Returns:
            True if this ingester can handle the file
        """
        return path.suffix.lower() in self.supported_extensions

    def ingest_directory(
        self,
        path: Path,
        tags: list[str] | None = None,
        recursive: bool = True,
    ) -> Iterator[IngestResult]:
        """Ingest all supported files in a directory.

        Args:
            path: Directory path
            tags: Optional tags to apply to all files
            recursive: Whether to recurse into subdirectories

        Yields:
            IngestResult for each file
        """
        pattern = "**/*" if recursive else "*"
        for file_path in path.glob(pattern):
            if file_path.is_file() and self.can_handle(file_path):
                try:
                    yield self.ingest_file(file_path, tags)
                except Exception:
                    # Skip files that fail to ingest
                    continue
