"""Tests for auto-detection ingester."""

from pathlib import Path

import pytest

from sidekick.core.models import SourceType
from sidekick.ingest.auto import AutoIngester, get_ingester


class TestAutoIngester:
    """Tests for AutoIngester."""

    def test_detect_markdown(self) -> None:
        """Test Markdown detection."""
        ingester = AutoIngester()

        assert ingester.detect_type(Path("file.md")) == SourceType.MARKDOWN
        assert ingester.detect_type(Path("file.markdown")) == SourceType.MARKDOWN

    def test_detect_json(self) -> None:
        """Test JSON detection."""
        ingester = AutoIngester()

        assert ingester.detect_type(Path("file.json")) == SourceType.JSON
        assert ingester.detect_type(Path("file.jsonl")) == SourceType.JSON

    def test_detect_text(self) -> None:
        """Test text detection."""
        ingester = AutoIngester()

        assert ingester.detect_type(Path("file.txt")) == SourceType.TEXT

    def test_unknown_falls_back_to_text(self) -> None:
        """Test unknown extensions fall back to text."""
        ingester = AutoIngester()

        assert ingester.detect_type(Path("file.xyz")) == SourceType.TEXT
        assert ingester.detect_type(Path("file.unknown")) == SourceType.TEXT

    def test_ingest_file_routes_correctly(self, temp_dir: Path) -> None:
        """Test that files are routed to correct ingester."""
        ingester = AutoIngester()

        # Create test files
        md_file = temp_dir / "test.md"
        md_file.write_text("# Hello\n\nWorld")

        json_file = temp_dir / "test.json"
        json_file.write_text('{"key": "value"}')

        txt_file = temp_dir / "test.txt"
        txt_file.write_text("Plain text")

        # Check routing
        md_result = ingester.ingest_file(md_file)
        assert md_result.source.source_type == SourceType.MARKDOWN

        json_result = ingester.ingest_file(json_file)
        assert json_result.source.source_type == SourceType.JSON

        txt_result = ingester.ingest_file(txt_file)
        assert txt_result.source.source_type == SourceType.TEXT

    def test_ingest_text_uses_text_ingester(self) -> None:
        """Test that raw text uses text ingester."""
        ingester = AutoIngester()
        result = ingester.ingest_text("Just some text")

        assert result.source.source_type == SourceType.TEXT

    def test_singleton(self) -> None:
        """Test that get_ingester returns singleton."""
        i1 = get_ingester()
        i2 = get_ingester()
        assert i1 is i2
