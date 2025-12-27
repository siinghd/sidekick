"""QA pair generation from text using LLMs."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from sidekick.core.models import QAPair
from sidekick.core.paths import get_paths
from sidekick.qa.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from sidekick.utils.fast_json import loads


@dataclass(slots=True)
class GenerationResult:
    """Result of QA generation. Uses __slots__ for memory efficiency."""

    qa_pairs: list[QAPair]
    source_id: str
    raw_response: str | None = None
    error: str | None = None


class QAGeneratorBackend(ABC):
    """Abstract backend for QA generation."""

    @abstractmethod
    def generate(self, text: str) -> str:
        """Generate QA pairs from text.

        Args:
            text: Source text to extract QA from

        Returns:
            Raw LLM response (should be JSON array)
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name for logging."""
        ...


class LlamaCppBackend(QAGeneratorBackend):
    """Local llama.cpp backend using GGUF models.

    Downloads and caches a small model for fully offline QA generation.
    """

    __slots__ = ("_model_path", "_repo_id", "_filename", "_n_ctx", "_n_gpu_layers", "_llm")

    # Default model - Qwen3-4B for best QA extraction quality
    # Using unsloth's GGUF conversion which is well-tested
    DEFAULT_MODEL_REPO = "unsloth/Qwen3-4B-GGUF"
    DEFAULT_MODEL_FILE = "Qwen3-4B-Q4_K_M.gguf"

    def __init__(
        self,
        model_path: Path | None = None,
        repo_id: str | None = None,
        filename: str | None = None,
        n_ctx: int = 8192,  # Qwen3 supports 32K+, use 8K for efficiency
        n_gpu_layers: int = -1,  # -1 = use all available
    ) -> None:
        """Initialize llama.cpp backend.

        Args:
            model_path: Path to GGUF model file (downloads if not provided)
            repo_id: HuggingFace repo to download from
            filename: Model filename in the repo
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
        """
        self._model_path = model_path
        self._repo_id = repo_id or self.DEFAULT_MODEL_REPO
        self._filename = filename or self.DEFAULT_MODEL_FILE
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._llm = None

    @property
    def name(self) -> str:
        return f"local/{self._filename}"

    def _ensure_model(self) -> Path:
        """Ensure model is downloaded and return path."""
        if self._model_path and self._model_path.exists():
            return self._model_path

        # Download to ~/.sidekick/models/base/
        from huggingface_hub import hf_hub_download

        model_dir = get_paths().base_models_dir
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = Path(
            hf_hub_download(
                repo_id=self._repo_id,
                filename=self._filename,
                local_dir=model_dir,
            )
        )

        self._model_path = model_path
        return model_path

    def _get_llm(self):
        """Get or create the LLM instance."""
        if self._llm is not None:
            return self._llm

        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required. Install with: "
                "pip install llama-cpp-python"
            )

        model_path = self._ensure_model()

        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=self._n_ctx,
            n_gpu_layers=self._n_gpu_layers,
            verbose=False,
        )

        return self._llm

    def generate(self, text: str) -> str:
        """Generate QA pairs using local model."""
        llm = self._get_llm()

        # Format as chat - add /nothink for Qwen3 to skip thinking mode
        prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{USER_PROMPT_TEMPLATE.format(text=text)} /nothink<|im_end|>
<|im_start|>assistant
"""

        response = llm(
            prompt,
            max_tokens=4000,  # Increased for Qwen3 which may use thinking tokens
            temperature=0.3,
            stop=["<|im_end|>", "<|im_start|>"],
        )

        result = response["choices"][0]["text"]

        # Strip Qwen3 thinking blocks if present
        if "<think>" in result:
            # Remove everything between <think> and </think>
            think_end = result.find("</think>")
            if think_end != -1:
                result = result[think_end + 8:].strip()
            else:
                # Thinking block not closed - try to find JSON after it
                json_start = result.find("[")
                if json_start != -1:
                    result = result[json_start:]

        return result


class OpenAIBackend(QAGeneratorBackend):
    """OpenAI API backend for QA generation."""

    __slots__ = ("_api_key", "_model", "_base_url")

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ) -> None:
        """Initialize OpenAI backend.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            model: Model to use
            base_url: Optional custom base URL (for compatible APIs)
        """
        self._api_key = api_key
        self._model = model
        self._base_url = base_url

    @property
    def name(self) -> str:
        return f"openai/{self._model}"

    def generate(self, text: str) -> str:
        """Generate QA pairs using OpenAI API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package is required. Install with: pip install openai"
            )

        import os

        api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY not set")

        client = OpenAI(api_key=api_key, base_url=self._base_url)

        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
            ],
            temperature=0.3,
            max_tokens=2000,
        )

        return response.choices[0].message.content or ""


class AnthropicBackend(QAGeneratorBackend):
    """Anthropic API backend for QA generation."""

    __slots__ = ("_api_key", "_model")

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-haiku-latest",
    ) -> None:
        """Initialize Anthropic backend.

        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
            model: Model to use
        """
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return f"anthropic/{self._model}"

    def generate(self, text: str) -> str:
        """Generate QA pairs using Anthropic API."""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package is required. Install with: pip install anthropic"
            )

        import os

        api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided and ANTHROPIC_API_KEY not set")

        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model=self._model,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
            ],
        )

        return response.content[0].text


class OllamaBackend(QAGeneratorBackend):
    """Ollama local model backend for QA generation."""

    __slots__ = ("_model", "_host")

    def __init__(
        self,
        model: str = "qwen3:4b",
        host: str = "http://localhost:11434",
    ) -> None:
        """Initialize Ollama backend.

        Args:
            model: Model name in Ollama (default: qwen3:4b)
            host: Ollama server URL
        """
        self._model = model
        self._host = host

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"

    def generate(self, text: str) -> str:
        """Generate QA pairs using Ollama."""
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx package is required. Install with: pip install httpx"
            )

        response = httpx.post(
            f"{self._host}/api/chat",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
                ],
                "stream": False,
                "options": {"temperature": 0.3},
            },
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


class QAGenerator:
    """Main QA generator that uses a backend to extract QA pairs from text."""

    __slots__ = ("_backend",)

    def __init__(self, backend: QAGeneratorBackend | None = None) -> None:
        """Initialize QA generator.

        Args:
            backend: LLM backend to use (auto-detects if not provided)
        """
        self._backend = backend or self._auto_detect_backend()

    def _auto_detect_backend(self) -> QAGeneratorBackend:
        """Auto-detect available backend.

        Priority:
        1. API keys (Anthropic, OpenAI) - fastest, best quality
        2. Ollama (if running) - good quality, no download needed
        3. Local llama.cpp - works offline, downloads model on first use
        """
        import os

        # Check for API keys first (best quality)
        if os.environ.get("ANTHROPIC_API_KEY"):
            return AnthropicBackend()
        if os.environ.get("OPENAI_API_KEY"):
            return OpenAIBackend()

        # Check for Ollama
        try:
            import httpx

            response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
            if response.status_code == 200:
                return OllamaBackend()
        except Exception:
            pass

        # Fall back to local llama.cpp (will download model on first use)
        return LlamaCppBackend()

    @property
    def backend_name(self) -> str:
        """Name of the current backend."""
        return self._backend.name

    def generate(self, text: str, source_id: str) -> GenerationResult:
        """Generate QA pairs from text.

        Args:
            text: Source text
            source_id: ID of the source document

        Returns:
            GenerationResult with extracted QA pairs
        """
        # For large texts, process in chunks
        # 8K context - 4K response = 4K input tokens ≈ 12K chars (conservative for JSON)
        max_chunk_chars = 12000

        if len(text) <= max_chunk_chars:
            chunks = [text]
        else:
            chunks = self._split_into_chunks(text, max_chunk_chars)

        all_qa_pairs: list[QAPair] = []
        all_responses: list[str] = []
        errors: list[str] = []

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            try:
                raw_response = self._backend.generate(chunk)
                all_responses.append(raw_response)
                qa_pairs = self._parse_response(raw_response, source_id)
                all_qa_pairs.extend(qa_pairs)
            except Exception as e:
                errors.append(f"Chunk {i+1}: {e}")

        return GenerationResult(
            qa_pairs=all_qa_pairs,
            source_id=source_id,
            raw_response="\n---\n".join(all_responses) if all_responses else None,
            error="; ".join(errors) if errors and not all_qa_pairs else None,
        )

    def _split_into_chunks(self, text: str, max_chars: int) -> list[str]:
        """Split text into chunks for processing.

        For JSONL content (one JSON per line), splits on line boundaries.
        For regular text, splits on paragraph or sentence boundaries.

        Args:
            text: Text to split
            max_chars: Maximum characters per chunk

        Returns:
            List of text chunks
        """
        lines = text.split("\n")

        # Check if this looks like JSONL (lines starting with {)
        is_jsonl = sum(1 for line in lines[:20] if line.strip().startswith("{")) > 10

        if is_jsonl:
            return self._chunk_jsonl(lines, max_chars)
        else:
            return self._chunk_text(text, max_chars)

    def _chunk_jsonl(self, lines: list[str], max_chars: int) -> list[str]:
        """Chunk JSONL content by grouping lines.

        Args:
            lines: Lines of JSONL content
            max_chars: Maximum chars per chunk

        Returns:
            List of chunks (each containing multiple JSONL lines)
        """
        chunks = []
        current_chunk: list[str] = []
        current_size = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            line_size = len(line) + 1  # +1 for newline

            if current_size + line_size > max_chars and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(line)
            current_size += line_size

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    def _chunk_text(self, text: str, max_chars: int) -> list[str]:
        """Chunk regular text by paragraphs.

        Args:
            text: Text to chunk
            max_chars: Maximum chars per chunk

        Returns:
            List of text chunks
        """
        # Split by double newlines (paragraphs)
        paragraphs = text.split("\n\n")

        chunks = []
        current_chunk: list[str] = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para) + 2  # +2 for \n\n

            if current_size + para_size > max_chars and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            # If single paragraph is too large, split by sentences
            if para_size > max_chars:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    if current_size + len(sent) > max_chars and current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                        current_chunk = []
                        current_size = 0
                    current_chunk.append(sent)
                    current_size += len(sent) + 1
            else:
                current_chunk.append(para)
                current_size += para_size

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _parse_response(self, response: str, source_id: str) -> list[QAPair]:
        """Parse LLM response into QA pairs.

        Args:
            response: Raw LLM response
            source_id: Source document ID

        Returns:
            List of QAPair objects
        """
        # Try to extract JSON from response
        response = response.strip()

        # Handle markdown code blocks
        if "```json" in response:
            match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
            if match:
                response = match.group(1)
        elif "```" in response:
            match = re.search(r"```\s*([\s\S]*?)\s*```", response)
            if match:
                response = match.group(1)

        # Try to find JSON array
        if not response.startswith("["):
            match = re.search(r"\[[\s\S]*\]", response)
            if match:
                response = match.group(0)

        try:
            data = loads(response)
        except (ValueError, TypeError):
            return []

        if not isinstance(data, list):
            return []

        qa_pairs: list[QAPair] = []
        for item in data:
            if not isinstance(item, dict):
                continue

            question = item.get("q") or item.get("question") or ""
            answer = item.get("a") or item.get("answer") or ""

            if question and answer:
                qa_pairs.append(
                    QAPair(
                        question=question.strip(),
                        answer=answer.strip(),
                        source_id=source_id,
                    )
                )

        return qa_pairs


def get_generator(backend: str | None = None) -> QAGenerator:
    """Get a QA generator with the specified backend.

    Args:
        backend: Backend name ("local", "openai", "anthropic", "ollama") or None for auto

    Returns:
        QAGenerator instance
    """
    if backend is None:
        return QAGenerator()

    backend_lower = backend.lower()
    if backend_lower == "local":
        return QAGenerator(LlamaCppBackend())
    elif backend_lower == "openai":
        return QAGenerator(OpenAIBackend())
    elif backend_lower == "anthropic":
        return QAGenerator(AnthropicBackend())
    elif backend_lower == "ollama":
        return QAGenerator(OllamaBackend())
    else:
        raise ValueError(f"Unknown backend: {backend}")
