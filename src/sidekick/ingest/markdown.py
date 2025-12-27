"""Markdown file ingester."""

import re
from pathlib import Path

from sidekick.core.models import Source, SourceType
from sidekick.ingest.base import BaseIngester, IngestResult
from sidekick.utils.hashing import hash_content, hash_file


class MarkdownIngester(BaseIngester):
    """Ingester for Markdown files.

    Extracts plain text from Markdown, preserving structure
    but removing formatting syntax.
    """

    @property
    def source_type(self) -> SourceType:
        return SourceType.MARKDOWN

    @property
    def supported_extensions(self) -> list[str]:
        return [".md", ".markdown", ".mdown", ".mkd"]

    def ingest_file(self, path: Path, tags: list[str] | None = None) -> IngestResult:
        """Ingest a Markdown file.

        Args:
            path: Path to the Markdown file
            tags: Optional tags

        Returns:
            IngestResult with source and extracted text
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = path.read_text(encoding="utf-8")
        content_hash = hash_file(path)
        text = self._extract_text(content)

        source = Source(
            id=content_hash[:16],
            path=path.absolute(),
            source_type=self.source_type,
            content_hash=content_hash,
            size_bytes=path.stat().st_size,
            tags=tags or [],
        )

        return IngestResult(
            source=source,
            text=text,
            metadata={"title": self._extract_title(content)},
        )

    def ingest_text(self, text: str, tags: list[str] | None = None) -> IngestResult:
        """Ingest raw Markdown text.

        Args:
            text: Markdown content
            tags: Optional tags

        Returns:
            IngestResult with source and extracted text
        """
        content_hash = hash_content(text)
        extracted = self._extract_text(text)

        source = Source(
            id=content_hash[:16],
            path=None,
            source_type=self.source_type,
            content_hash=content_hash,
            size_bytes=len(text.encode("utf-8")),
            tags=tags or [],
        )

        return IngestResult(
            source=source,
            text=extracted,
            metadata={"title": self._extract_title(text)},
        )

    def _extract_text(self, markdown: str) -> str:
        """Extract plain text from Markdown.

        Args:
            markdown: Markdown content

        Returns:
            Plain text with formatting removed
        """
        text = markdown

        # Remove code blocks (preserve content for context)
        text = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).split("\n", 1)[-1].rsplit("```", 1)[0], text)
        text = re.sub(r"`([^`]+)`", r"\1", text)

        # Remove images but keep alt text
        text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)

        # Convert links to just text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # Remove emphasis markers
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)
        text = re.sub(r"~~([^~]+)~~", r"\1", text)

        # Convert headers to plain text (keep the text)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

        # Remove horizontal rules
        text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

        # Clean up list markers
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

        # Remove blockquote markers
        text = re.sub(r"^\s*>\s*", "", text, flags=re.MULTILINE)

        # Clean up extra whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        return text

    def _extract_title(self, markdown: str) -> str | None:
        """Extract the title (first H1) from Markdown.

        Args:
            markdown: Markdown content

        Returns:
            Title text or None
        """
        match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return None
