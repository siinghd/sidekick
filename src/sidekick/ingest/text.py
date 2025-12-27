"""Text file ingester."""

from pathlib import Path

from sidekick.core.models import Source, SourceType
from sidekick.ingest.base import BaseIngester, IngestResult
from sidekick.utils.hashing import hash_content, hash_file


class TextIngester(BaseIngester):
    """Ingester for plain text files."""

    @property
    def source_type(self) -> SourceType:
        return SourceType.TEXT

    @property
    def supported_extensions(self) -> list[str]:
        return [".txt", ".text", ""]

    def ingest_file(self, path: Path, tags: list[str] | None = None) -> IngestResult:
        """Ingest a text file.

        Args:
            path: Path to the text file
            tags: Optional tags

        Returns:
            IngestResult with source and text content
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = path.read_text(encoding="utf-8")
        content_hash = hash_file(path)

        source = Source(
            id=content_hash[:16],
            path=path.absolute(),
            source_type=self.source_type,
            content_hash=content_hash,
            size_bytes=path.stat().st_size,
            tags=tags or [],
        )

        return IngestResult(source=source, text=content)

    def ingest_text(self, text: str, tags: list[str] | None = None) -> IngestResult:
        """Ingest raw text.

        Args:
            text: Text content
            tags: Optional tags

        Returns:
            IngestResult with source and text content
        """
        content_hash = hash_content(text)

        source = Source(
            id=content_hash[:16],
            path=None,
            source_type=self.source_type,
            content_hash=content_hash,
            size_bytes=len(text.encode("utf-8")),
            tags=tags or [],
        )

        return IngestResult(source=source, text=text)
