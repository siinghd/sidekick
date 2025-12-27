"""CSV file ingester."""

import csv
import io
from pathlib import Path

from sidekick.core.models import Source, SourceType
from sidekick.ingest.base import BaseIngester, IngestResult
from sidekick.utils.hashing import hash_content, hash_file


class CsvIngester(BaseIngester):
    """Ingester for CSV files.

    Converts CSV data to readable text format.
    """

    @property
    def source_type(self) -> SourceType:
        return SourceType.CSV

    @property
    def supported_extensions(self) -> list[str]:
        return [".csv", ".tsv"]

    def ingest_file(self, path: Path, tags: list[str] | None = None) -> IngestResult:
        """Ingest a CSV file.

        Args:
            path: Path to the CSV file
            tags: Optional tags

        Returns:
            IngestResult with source and extracted text
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = path.read_text(encoding="utf-8", errors="ignore")
        content_hash = hash_file(path)

        delimiter = "\t" if path.suffix == ".tsv" else ","
        text, row_count = self._extract_text(content, delimiter)

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
            metadata={"row_count": row_count},
        )

    def ingest_text(self, text: str, tags: list[str] | None = None) -> IngestResult:
        """Ingest raw CSV text.

        Args:
            text: CSV content
            tags: Optional tags

        Returns:
            IngestResult with source and extracted text
        """
        content_hash = hash_content(text)
        extracted, row_count = self._extract_text(text, ",")

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
            metadata={"row_count": row_count},
        )

    def _extract_text(self, content: str, delimiter: str = ",") -> tuple[str, int]:
        """Extract readable text from CSV.

        Args:
            content: CSV content
            delimiter: Field delimiter

        Returns:
            Tuple of (text, row_count)
        """
        rows: list[str] = []
        headers: list[str] = []

        try:
            reader = csv.reader(io.StringIO(content), delimiter=delimiter)
            for i, row in enumerate(reader):
                if i == 0:
                    # First row is headers
                    headers = [h.strip() for h in row]
                    continue

                # Format row as "header: value" pairs
                parts = []
                for j, cell in enumerate(row):
                    cell = cell.strip()
                    if not cell:
                        continue
                    if j < len(headers) and headers[j]:
                        parts.append(f"{headers[j]}: {cell}")
                    else:
                        parts.append(cell)

                if parts:
                    rows.append(", ".join(parts))

        except csv.Error:
            # If CSV parsing fails, just return the raw content
            return content, 0

        text = "\n".join(rows)
        return text, len(rows)
