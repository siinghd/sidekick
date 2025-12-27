"""Tests for data ingesters."""

from pathlib import Path

import pytest

from sidekick.core.models import SourceType
from sidekick.ingest.json_ingester import JsonIngester
from sidekick.ingest.markdown import MarkdownIngester
from sidekick.ingest.text import TextIngester


class TestTextIngester:
    """Tests for TextIngester."""

    def test_ingest_text(self) -> None:
        """Test ingesting raw text."""
        ingester = TextIngester()
        result = ingester.ingest_text("Hello, world!")

        assert result.text == "Hello, world!"
        assert result.source.source_type == SourceType.TEXT
        assert result.source.size_bytes == len("Hello, world!".encode())

    def test_ingest_text_with_tags(self) -> None:
        """Test ingesting with tags."""
        ingester = TextIngester()
        result = ingester.ingest_text("Test content", tags=["work", "notes"])

        assert result.source.tags == ["work", "notes"]

    def test_ingest_file(self, temp_dir: Path) -> None:
        """Test ingesting a text file."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("File content here")

        ingester = TextIngester()
        result = ingester.ingest_file(test_file)

        assert result.text == "File content here"
        assert result.source.path == test_file.absolute()
        assert result.source.source_type == SourceType.TEXT

    def test_can_handle(self, temp_dir: Path) -> None:
        """Test file type detection."""
        ingester = TextIngester()

        assert ingester.can_handle(Path("file.txt"))
        assert ingester.can_handle(Path("file.text"))
        assert not ingester.can_handle(Path("file.md"))
        assert not ingester.can_handle(Path("file.json"))


class TestMarkdownIngester:
    """Tests for MarkdownIngester."""

    def test_ingest_markdown(self) -> None:
        """Test ingesting Markdown."""
        ingester = MarkdownIngester()
        md = """# Title

This is **bold** and *italic* text.

- List item 1
- List item 2

[A link](https://example.com)
"""
        result = ingester.ingest_text(md)

        assert "Title" in result.text
        assert "bold" in result.text
        assert "italic" in result.text
        assert "**" not in result.text
        assert "*" not in result.text
        assert "[A link]" not in result.text
        assert "A link" in result.text

    def test_extract_title(self) -> None:
        """Test title extraction."""
        ingester = MarkdownIngester()
        result = ingester.ingest_text("# My Document\n\nContent here")

        assert result.metadata is not None
        assert result.metadata.get("title") == "My Document"

    def test_code_blocks(self) -> None:
        """Test code block handling."""
        ingester = MarkdownIngester()
        md = """Some text

```python
def hello():
    print("Hello")
```

More text
"""
        result = ingester.ingest_text(md)

        assert "Some text" in result.text
        assert "More text" in result.text
        # Code content should be preserved
        assert "hello" in result.text.lower()


class TestJsonIngester:
    """Tests for JsonIngester."""

    def test_simple_object(self) -> None:
        """Test ingesting a simple JSON object."""
        ingester = JsonIngester()
        result = ingester.ingest_text('{"name": "John", "job": "Engineer"}')

        assert "John" in result.text
        assert "Engineer" in result.text

    def test_chat_format(self) -> None:
        """Test ingesting chat/messages format."""
        ingester = JsonIngester()
        chat = """[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]"""
        result = ingester.ingest_text(chat)

        assert "user: Hello" in result.text
        assert "assistant: Hi there!" in result.text

    def test_jsonl(self, temp_dir: Path) -> None:
        """Test ingesting JSONL file."""
        jsonl_file = temp_dir / "data.jsonl"
        jsonl_file.write_text(
            '{"text": "First line"}\n'
            '{"text": "Second line"}\n'
        )

        ingester = JsonIngester()
        result = ingester.ingest_file(jsonl_file)

        assert "First line" in result.text
        assert "Second line" in result.text

    def test_nested_messages(self) -> None:
        """Test conversation wrapper format."""
        ingester = JsonIngester()
        data = """{
            "id": "conv-123",
            "messages": [
                {"role": "user", "content": "What time is it?"},
                {"role": "assistant", "content": "It's noon."}
            ]
        }"""
        result = ingester.ingest_text(data)

        assert "What time is it?" in result.text
        assert "It's noon" in result.text
