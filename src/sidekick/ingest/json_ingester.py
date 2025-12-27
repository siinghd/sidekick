"""JSON file ingester."""

import json
import os
from pathlib import Path
from typing import Any

from sidekick.core.models import Source, SourceType
from sidekick.ingest.base import BaseIngester, IngestResult
from sidekick.utils.hashing import hash_content, hash_file


def _ask_llm_for_fields(sample_json: str, verbose: bool = True) -> tuple[list[str], str]:
    """Ask LLM which fields contain relevant content.

    Args:
        sample_json: Sample of JSON data to analyze
        verbose: Whether to print status

    Returns:
        Tuple of (field paths, backend used)
    """
    prompt = f"""Analyze this JSON sample and identify which fields contain meaningful text content
(conversations, messages, notes, documents - NOT metadata like IDs, timestamps, paths, tokens, etc).

JSON sample:
{sample_json[:3000]}

Return ONLY a JSON array of dot-notation field paths to extract, e.g.:
["message.content", "text", "body"]

If content is in an array, use [] notation, e.g. "messages[].content"
Return empty array [] if no meaningful text fields found. /nothink"""

    def _parse_response(text: str) -> list[str]:
        """Extract JSON array from response."""
        # Strip thinking blocks
        if "<think>" in text:
            end = text.find("</think>")
            if end != -1:
                text = text[end + 8:]
        # Find JSON array
        if "[" in text:
            start = text.find("[")
            end = text.rfind("]") + 1
            text = text[start:end]
        return json.loads(text)

    # Try Ollama first (local, fast)
    try:
        import httpx
        response = httpx.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "qwen3:4b",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=30.0,
        )
        if response.status_code == 200:
            text = response.json()["message"]["content"]
            fields = _parse_response(text)
            return fields, "ollama/qwen3:4b"
    except Exception as e:
        if verbose:
            print(f"  [dim]Ollama: {e}[/dim]")

    # Try local llama.cpp
    try:
        from llama_cpp import Llama
        from sidekick.core.paths import get_paths

        model_dir = get_paths().base_models_dir
        model_file = model_dir / "Qwen3-4B-Q4_K_M.gguf"

        if model_file.exists():
            llm = Llama(
                model_path=str(model_file),
                n_ctx=4096,
                n_gpu_layers=-1,
                verbose=False,
            )
            formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            response = llm(formatted, max_tokens=200, stop=["<|im_end|>"])
            text = response["choices"][0]["text"]
            fields = _parse_response(text)
            return fields, "llama.cpp/qwen3"
    except Exception as e:
        if verbose:
            print(f"  [dim]llama.cpp: {e}[/dim]")

    # Try API backends
    try:
        import anthropic
        if os.environ.get("ANTHROPIC_API_KEY"):
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            fields = _parse_response(response.content[0].text)
            return fields, "anthropic/haiku"
    except Exception as e:
        if verbose:
            print(f"  [dim]Anthropic: {e}[/dim]")

    try:
        from openai import OpenAI
        if os.environ.get("OPENAI_API_KEY"):
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            fields = _parse_response(response.choices[0].message.content or "[]")
            return fields, "openai/gpt-4o-mini"
    except Exception as e:
        if verbose:
            print(f"  [dim]OpenAI: {e}[/dim]")

    # Fallback to common patterns
    return ["message.content", "content", "text", "body"], "fallback"


def _extract_by_path(data: Any, path: str) -> list[str]:
    """Extract values from data using dot notation path.

    Args:
        data: JSON data
        path: Dot notation path like "message.content" or "messages[].content"

    Returns:
        List of extracted string values
    """
    results = []
    parts = path.replace("[]", ".[*]").split(".")

    def _traverse(obj: Any, remaining: list[str]) -> None:
        if not remaining:
            if isinstance(obj, str) and obj.strip():
                results.append(obj.strip())
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict) and "text" in item:
                        results.append(item["text"])
                    elif isinstance(item, str):
                        results.append(item)
            return

        key = remaining[0]
        rest = remaining[1:]

        if key == "[*]":
            if isinstance(obj, list):
                for item in obj:
                    _traverse(item, rest)
        elif isinstance(obj, dict) and key in obj:
            _traverse(obj[key], rest)

    _traverse(data, parts)
    return results


class JsonIngester(BaseIngester):
    """Ingester for JSON files.

    Handles various JSON formats including:
    - Plain JSON objects/arrays
    - Chat exports (OpenAI, Claude format)
    - JSONL files (one JSON object per line)
    """

    @property
    def source_type(self) -> SourceType:
        return SourceType.JSON

    @property
    def supported_extensions(self) -> list[str]:
        return [".json", ".jsonl"]

    def ingest_file(self, path: Path, tags: list[str] | None = None) -> IngestResult:
        """Ingest a JSON file.

        Args:
            path: Path to the JSON file
            tags: Optional tags

        Returns:
            IngestResult with source and extracted text
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = path.read_text(encoding="utf-8")
        content_hash = hash_file(path)

        # Get sample for LLM analysis
        if path.suffix == ".jsonl":
            lines = content.split("\n")[:10]  # First 10 lines
            sample = "\n".join(lines)
        else:
            sample = content[:3000]

        # Ask LLM which fields to extract
        field_paths, backend = _ask_llm_for_fields(sample)
        print(f"  Field detection: {backend} → {field_paths}")

        # Extract content using detected field paths
        text = self._extract_with_paths(content, field_paths, is_jsonl=(path.suffix == ".jsonl"))
        print(f"  Extracted: {len(text):,} chars from {len(content):,} chars ({100*len(text)//len(content)}%)")

        # Fallback to old method if nothing extracted
        if not text.strip():
            if path.suffix == ".jsonl":
                text = self._extract_jsonl(content)
            else:
                try:
                    data = json.loads(content)
                    text = self._extract_text(data)
                except json.JSONDecodeError:
                    text = self._extract_jsonl(content)

        source = Source(
            id=content_hash[:16],
            path=path.absolute(),
            source_type=self.source_type,
            content_hash=content_hash,
            size_bytes=path.stat().st_size,
            tags=tags or [],
        )

        return IngestResult(source=source, text=text)

    def _extract_with_paths(self, content: str, paths: list[str], is_jsonl: bool) -> str:
        """Extract content using field paths.

        Args:
            content: Raw JSON/JSONL content
            paths: List of field paths to extract
            is_jsonl: Whether content is JSONL format

        Returns:
            Extracted text
        """
        all_texts = []

        if is_jsonl:
            for line in content.split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    for path in paths:
                        texts = _extract_by_path(data, path)
                        all_texts.extend(texts)
                except json.JSONDecodeError:
                    continue
        else:
            try:
                data = json.loads(content)
                for path in paths:
                    texts = _extract_by_path(data, path)
                    all_texts.extend(texts)
            except json.JSONDecodeError:
                pass

        return "\n\n".join(all_texts)

    def ingest_text(self, text: str, tags: list[str] | None = None) -> IngestResult:
        """Ingest raw JSON text.

        Args:
            text: JSON content
            tags: Optional tags

        Returns:
            IngestResult with source and extracted text
        """
        content_hash = hash_content(text)

        # Try to parse as JSON
        try:
            data = json.loads(text)
            extracted = self._extract_text(data)
        except json.JSONDecodeError:
            # Try as JSONL
            extracted = self._extract_jsonl(text)

        source = Source(
            id=content_hash[:16],
            path=None,
            source_type=self.source_type,
            content_hash=content_hash,
            size_bytes=len(text.encode("utf-8")),
            tags=tags or [],
        )

        return IngestResult(source=source, text=extracted)

    def _extract_text(self, data: Any, depth: int = 0) -> str:
        """Recursively extract text from JSON data.

        Args:
            data: JSON data (dict, list, or primitive)
            depth: Current recursion depth

        Returns:
            Extracted text
        """
        if depth > 20:  # Prevent infinite recursion
            return ""

        if isinstance(data, str):
            return data

        if isinstance(data, (int, float, bool)):
            return str(data)

        if isinstance(data, list):
            # Check for chat format
            if self._is_chat_format(data):
                return self._extract_chat(data)
            # Regular list
            parts = [self._extract_text(item, depth + 1) for item in data]
            return "\n".join(p for p in parts if p)

        if isinstance(data, dict):
            # Check for direct message format: {"role": "user", "content": "..."}
            if "content" in data and "role" in data:
                role = data.get("role", "")
                content = data.get("content", "")
                # Handle array content (Claude API format)
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and "text" in p
                    )
                return f"{role}: {content}" if content else ""

            # Check for nested message format: {"type": "...", "message": {"role": "...", "content": "..."}}
            # Common in conversation logs, chat exports, etc.
            if "message" in data and isinstance(data["message"], dict):
                msg = data["message"]
                msg_type = data.get("type", "")
                # Skip pure metadata entries
                if msg_type in ("file-history-snapshot", "summary", "system-info", "metadata"):
                    return ""
                # Skip meta/system messages
                if data.get("isMeta") or data.get("isSystem"):
                    return ""
                if "content" in msg:
                    role = msg.get("role", msg_type or "unknown")
                    content = msg.get("content", "")
                    # Handle array content
                    if isinstance(content, list):
                        content = " ".join(
                            p.get("text", "") for p in content
                            if isinstance(p, dict) and "text" in p
                        )
                    # Skip XML-like command messages
                    if isinstance(content, str) and content.strip().startswith("<command"):
                        return ""
                    if content:
                        return f"{role}: {content}"
                return ""

            # Check for conversation wrapper: {"messages": [...]}
            if "messages" in data and isinstance(data["messages"], list):
                return self._extract_chat(data["messages"])

            # Generic dict - extract text values, skip common metadata
            SKIP_KEYS = {
                "id", "uuid", "parentUuid", "messageId",
                "created_at", "updated_at", "timestamp", "date",
                "sessionId", "conversationId", "threadId",
                "version", "cwd", "path", "gitBranch",
                "userType", "type", "model", "provider",
                "isSidechain", "isMeta", "isSystem", "isSnapshotUpdate",
                "snapshot", "trackedFileBackups", "metadata",
                "tokens", "usage", "cost", "latency",
            }
            parts = []
            for key, value in data.items():
                if key.lower() in SKIP_KEYS or key.startswith("_"):
                    continue
                extracted = self._extract_text(value, depth + 1)
                if extracted:
                    parts.append(extracted)
            return "\n".join(parts)

        return ""

    def _is_chat_format(self, data: list[Any]) -> bool:
        """Check if data is a chat/messages format.

        Args:
            data: List to check

        Returns:
            True if it looks like a chat format
        """
        if not data:
            return False
        # Check first few items for message structure
        sample = data[:3]
        return all(
            isinstance(item, dict) and "role" in item and "content" in item
            for item in sample
        )

    def _extract_chat(self, messages: list[dict[str, Any]]) -> str:
        """Extract text from a chat/messages format.

        Args:
            messages: List of message dicts

        Returns:
            Formatted conversation text
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle multi-part content (like Claude's format)
                text_parts = [
                    p.get("text", "") for p in content if isinstance(p, dict) and "text" in p
                ]
                content = " ".join(text_parts)
            if content:
                parts.append(f"{role}: {content}")
        return "\n\n".join(parts)

    def _extract_jsonl(self, content: str) -> str:
        """Extract text from JSONL content.

        Args:
            content: JSONL content (one JSON per line)

        Returns:
            Combined extracted text
        """
        parts = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                extracted = self._extract_text(data)
                if extracted:
                    parts.append(extracted)
            except json.JSONDecodeError:
                continue
        return "\n\n".join(parts)
