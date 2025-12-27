"""Tests for PDF, HTML, CSV, and mbox ingesters."""

from pathlib import Path

import pytest

from sidekick.core.models import SourceType
from sidekick.ingest.csv_ingester import CsvIngester
from sidekick.ingest.html import HtmlIngester
from sidekick.ingest.mbox import MboxIngester
from sidekick.ingest.pdf import PdfIngester


class TestHtmlIngester:
    """Tests for HtmlIngester."""

    def test_extract_text(self) -> None:
        """Test extracting text from HTML."""
        ingester = HtmlIngester()
        html = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Hello World</h1>
            <p>This is a <strong>test</strong> paragraph.</p>
            <script>var x = 1;</script>
        </body>
        </html>
        """
        result = ingester.ingest_text(html)

        assert "Hello World" in result.text
        assert "test" in result.text
        assert "paragraph" in result.text
        assert "var x = 1" not in result.text  # Script should be stripped
        assert result.source.source_type == SourceType.HTML

    def test_extract_title(self) -> None:
        """Test title extraction."""
        ingester = HtmlIngester()
        html = "<html><head><title>My Title</title></head><body>Content</body></html>"
        result = ingester.ingest_text(html)

        assert result.metadata is not None
        assert result.metadata.get("title") == "My Title"

    def test_ingest_file(self, temp_dir: Path) -> None:
        """Test ingesting HTML file."""
        html_file = temp_dir / "test.html"
        html_file.write_text("<html><body><p>Hello</p></body></html>")

        ingester = HtmlIngester()
        result = ingester.ingest_file(html_file)

        assert "Hello" in result.text
        assert result.source.path == html_file.absolute()


class TestCsvIngester:
    """Tests for CsvIngester."""

    def test_simple_csv(self) -> None:
        """Test ingesting simple CSV."""
        ingester = CsvIngester()
        csv = """name,age,city
John,30,NYC
Jane,25,LA"""
        result = ingester.ingest_text(csv)

        assert "name: John" in result.text
        assert "age: 30" in result.text
        assert "city: NYC" in result.text
        assert result.source.source_type == SourceType.CSV

    def test_csv_file(self, temp_dir: Path) -> None:
        """Test ingesting CSV file."""
        csv_file = temp_dir / "data.csv"
        csv_file.write_text("col1,col2\nval1,val2\n")

        ingester = CsvIngester()
        result = ingester.ingest_file(csv_file)

        assert "col1: val1" in result.text
        assert "col2: val2" in result.text
        assert result.metadata is not None
        assert result.metadata.get("row_count") == 1

    def test_tsv_file(self, temp_dir: Path) -> None:
        """Test ingesting TSV file."""
        tsv_file = temp_dir / "data.tsv"
        tsv_file.write_text("name\tscore\nAlice\t100\n")

        ingester = CsvIngester()
        result = ingester.ingest_file(tsv_file)

        assert "name: Alice" in result.text
        assert "score: 100" in result.text


class TestMboxIngester:
    """Tests for MboxIngester."""

    def test_ingest_mbox_file(self, temp_dir: Path) -> None:
        """Test ingesting mbox file."""
        mbox_file = temp_dir / "test.mbox"
        mbox_content = """From sender@example.com Mon Jan 01 00:00:00 2024
From: sender@example.com
To: recipient@example.com
Subject: Test Email
Date: Mon, 01 Jan 2024 00:00:00 +0000
Content-Type: text/plain

This is the email body.

From another@example.com Mon Jan 02 00:00:00 2024
From: another@example.com
To: someone@example.com
Subject: Second Email
Date: Tue, 02 Jan 2024 00:00:00 +0000
Content-Type: text/plain

Second email content.
"""
        mbox_file.write_text(mbox_content)

        ingester = MboxIngester()
        result = ingester.ingest_file(mbox_file)

        assert "Test Email" in result.text
        assert "This is the email body" in result.text
        assert "Second Email" in result.text
        assert result.source.source_type == SourceType.MBOX
        assert result.metadata is not None
        assert result.metadata.get("message_count") == 2

    def test_ingest_text_raises(self) -> None:
        """Test that ingest_text raises NotImplementedError."""
        ingester = MboxIngester()
        with pytest.raises(NotImplementedError):
            ingester.ingest_text("not supported")


class TestPdfIngester:
    """Tests for PdfIngester."""

    def test_ingest_text_raises(self) -> None:
        """Test that ingest_text raises NotImplementedError."""
        ingester = PdfIngester()
        with pytest.raises(NotImplementedError):
            ingester.ingest_text("not supported")

    def test_source_type(self) -> None:
        """Test source type is PDF."""
        ingester = PdfIngester()
        assert ingester.source_type == SourceType.PDF

    def test_supported_extensions(self) -> None:
        """Test supported extensions."""
        ingester = PdfIngester()
        assert ".pdf" in ingester.supported_extensions
