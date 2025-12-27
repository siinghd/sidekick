"""HTML file ingester."""

import re
from html.parser import HTMLParser
from pathlib import Path

from sidekick.core.models import Source, SourceType
from sidekick.ingest.base import BaseIngester, IngestResult
from sidekick.utils.hashing import hash_content, hash_file


class _HTMLTextExtractor(HTMLParser):
    """HTML parser that extracts text content."""

    def __init__(self) -> None:
        super().__init__()
        self.text_parts: list[str] = []
        self.title: str = ""
        self._in_title = False
        self._skip_tags = {"script", "style", "meta", "link", "noscript"}
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._skip_tags:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in ("br", "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li"):
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._skip_tags:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6"):
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_title:
            self.title = data.strip()
        text = data.strip()
        if text:
            self.text_parts.append(text)

    def get_text(self) -> str:
        """Get extracted text, cleaned up."""
        text = " ".join(self.text_parts)
        # Clean up whitespace
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = re.sub(r" +", " ", text)
        return text.strip()


class HtmlIngester(BaseIngester):
    """Ingester for HTML files.

    Extracts text content, stripping tags and scripts.
    """

    @property
    def source_type(self) -> SourceType:
        return SourceType.HTML

    @property
    def supported_extensions(self) -> list[str]:
        return [".html", ".htm", ".xhtml"]

    def ingest_file(self, path: Path, tags: list[str] | None = None) -> IngestResult:
        """Ingest an HTML file.

        Args:
            path: Path to the HTML file
            tags: Optional tags

        Returns:
            IngestResult with source and extracted text
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = path.read_text(encoding="utf-8", errors="ignore")
        content_hash = hash_file(path)

        text, title = self._extract_text(content)

        source = Source(
            id=content_hash[:16],
            path=path.absolute(),
            source_type=self.source_type,
            content_hash=content_hash,
            size_bytes=path.stat().st_size,
            tags=tags or [],
        )

        metadata: dict[str, str | int | float | bool] = {}
        if title:
            metadata["title"] = title

        return IngestResult(source=source, text=text, metadata=metadata)

    def ingest_text(self, text: str, tags: list[str] | None = None) -> IngestResult:
        """Ingest raw HTML text.

        Args:
            text: HTML content
            tags: Optional tags

        Returns:
            IngestResult with source and extracted text
        """
        content_hash = hash_content(text)
        extracted, title = self._extract_text(text)

        source = Source(
            id=content_hash[:16],
            path=None,
            source_type=self.source_type,
            content_hash=content_hash,
            size_bytes=len(text.encode("utf-8")),
            tags=tags or [],
        )

        metadata: dict[str, str | int | float | bool] = {}
        if title:
            metadata["title"] = title

        return IngestResult(source=source, text=extracted, metadata=metadata)

    def _extract_text(self, html: str) -> tuple[str, str]:
        """Extract text from HTML.

        Args:
            html: HTML content

        Returns:
            Tuple of (text, title)
        """
        parser = _HTMLTextExtractor()
        try:
            parser.feed(html)
        except Exception:
            # If parsing fails, just strip tags naively
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text)
            return text.strip(), ""

        return parser.get_text(), parser.title
