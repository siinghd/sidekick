"""PDF file ingester."""

from pathlib import Path

from sidekick.core.models import Source, SourceType
from sidekick.ingest.base import BaseIngester, IngestResult
from sidekick.utils.hashing import hash_file


class PdfIngester(BaseIngester):
    """Ingester for PDF files.

    Uses PyMuPDF (fitz) for text extraction.
    """

    @property
    def source_type(self) -> SourceType:
        return SourceType.PDF

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def ingest_file(self, path: Path, tags: list[str] | None = None) -> IngestResult:
        """Ingest a PDF file.

        Args:
            path: Path to the PDF file
            tags: Optional tags

        Returns:
            IngestResult with source and extracted text
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF ingestion. "
                "Install with: pip install pymupdf"
            )

        content_hash = hash_file(path)
        text_parts: list[str] = []
        metadata: dict[str, str | int | float | bool] = {}

        with fitz.open(path) as doc:
            metadata["page_count"] = len(doc)
            metadata["title"] = doc.metadata.get("title", "") or ""
            metadata["author"] = doc.metadata.get("author", "") or ""

            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text("text")
                if page_text.strip():
                    text_parts.append(f"[Page {page_num}]\n{page_text}")

        text = "\n\n".join(text_parts)

        source = Source(
            id=content_hash[:16],
            path=path.absolute(),
            source_type=self.source_type,
            content_hash=content_hash,
            size_bytes=path.stat().st_size,
            tags=tags or [],
        )

        return IngestResult(source=source, text=text, metadata=metadata)

    def ingest_text(self, text: str, tags: list[str] | None = None) -> IngestResult:
        """PDF ingester doesn't support raw text input.

        Raises:
            NotImplementedError: Always
        """
        raise NotImplementedError("PDF ingester requires a file path")
