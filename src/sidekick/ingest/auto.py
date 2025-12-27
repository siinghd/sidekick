"""Auto-detection and routing for file ingestion."""

from pathlib import Path
from typing import Iterator

from sidekick.core.models import SourceType
from sidekick.ingest.base import BaseIngester, IngestResult
from sidekick.ingest.csv_ingester import CsvIngester
from sidekick.ingest.html import HtmlIngester
from sidekick.ingest.json_ingester import JsonIngester
from sidekick.ingest.markdown import MarkdownIngester
from sidekick.ingest.mbox import MboxIngester
from sidekick.ingest.pdf import PdfIngester
from sidekick.ingest.text import TextIngester


class AutoIngester:
    """Auto-detecting ingester that routes to the appropriate handler.

    Automatically detects file types and delegates to the correct
    specialized ingester.

    Supported formats:
    - Markdown (.md, .markdown)
    - JSON/JSONL (.json, .jsonl)
    - PDF (.pdf)
    - HTML (.html, .htm)
    - CSV/TSV (.csv, .tsv)
    - Mbox (.mbox, .mbx)
    - Text (.txt, and fallback for unknown)
    """

    def __init__(self) -> None:
        """Initialize with all available ingesters."""
        self._ingesters: list[BaseIngester] = [
            MarkdownIngester(),
            JsonIngester(),
            PdfIngester(),
            HtmlIngester(),
            CsvIngester(),
            MboxIngester(),
            TextIngester(),  # Fallback - handles .txt and unknown
        ]

        # Build extension -> ingester mapping
        self._extension_map: dict[str, BaseIngester] = {}
        for ingester in self._ingesters:
            for ext in ingester.supported_extensions:
                if ext not in self._extension_map:
                    self._extension_map[ext] = ingester

    def get_ingester(self, path: Path) -> BaseIngester:
        """Get the appropriate ingester for a file.

        Args:
            path: Path to the file

        Returns:
            The ingester that can handle this file type
        """
        ext = path.suffix.lower()
        if ext in self._extension_map:
            return self._extension_map[ext]
        # Fallback to text ingester for unknown types
        return self._ingesters[-1]

    def detect_type(self, path: Path) -> SourceType:
        """Detect the source type for a file.

        Args:
            path: Path to the file

        Returns:
            The detected SourceType
        """
        ingester = self.get_ingester(path)
        return ingester.source_type

    def ingest_file(self, path: Path, tags: list[str] | None = None) -> IngestResult:
        """Ingest a file, auto-detecting the type.

        Args:
            path: Path to the file
            tags: Optional tags

        Returns:
            IngestResult from the appropriate ingester
        """
        ingester = self.get_ingester(path)
        return ingester.ingest_file(path, tags)

    def ingest_text(self, text: str, tags: list[str] | None = None) -> IngestResult:
        """Ingest raw text using the text ingester.

        Args:
            text: Raw text
            tags: Optional tags

        Returns:
            IngestResult
        """
        # Raw text always goes to text ingester
        return TextIngester().ingest_text(text, tags)

    def ingest_directory(
        self,
        path: Path,
        tags: list[str] | None = None,
        recursive: bool = True,
    ) -> Iterator[IngestResult]:
        """Ingest all supported files in a directory.

        Args:
            path: Directory path
            tags: Optional tags
            recursive: Whether to recurse

        Yields:
            IngestResult for each successfully ingested file
        """
        pattern = "**/*" if recursive else "*"
        for file_path in sorted(path.glob(pattern)):
            if file_path.is_file() and not file_path.name.startswith("."):
                try:
                    yield self.ingest_file(file_path, tags)
                except Exception:
                    # Skip files that fail to ingest
                    continue

    @property
    def supported_extensions(self) -> list[str]:
        """Get all supported file extensions."""
        return list(self._extension_map.keys())


# Singleton instance
_auto_ingester: AutoIngester | None = None


def get_ingester() -> AutoIngester:
    """Get the global AutoIngester instance."""
    global _auto_ingester
    if _auto_ingester is None:
        _auto_ingester = AutoIngester()
    return _auto_ingester
