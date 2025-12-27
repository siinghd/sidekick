"""Ollama inference backend."""

import time
import urllib.request
import urllib.error
from typing import Any

from sidekick.inference.base import (
    BaseInferenceBackend,
    InferenceConfig,
    InferenceResult,
)
from sidekick.utils.fast_json import dumps, loads


class OllamaBackend(BaseInferenceBackend):
    """Inference backend using Ollama.

    This backend connects to a local Ollama instance for inference.
    Supports any model available in Ollama.
    """

    __slots__ = ("model", "base_url")

    DEFAULT_MODEL = "qwen2.5:1.5b"
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        config: InferenceConfig,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize the Ollama backend.

        Args:
            config: Inference configuration
            model: Ollama model name (default: qwen2.5:1.5b)
            base_url: Ollama API base URL (default: http://localhost:11434)
        """
        super().__init__(config)
        self.model = model or self.DEFAULT_MODEL
        self.base_url = base_url or self.DEFAULT_BASE_URL

    @property
    def name(self) -> str:
        return "ollama"

    def is_available(self) -> tuple[bool, str]:
        """Check if Ollama is available."""
        try:
            url = f"{self.base_url}/api/tags"
            request = urllib.request.Request(url, method="GET")

            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status == 200:
                    return True, "Ollama is available"
                return False, f"Ollama returned status {response.status}"

        except urllib.error.URLError as e:
            return False, f"Cannot connect to Ollama: {e}"
        except Exception as e:
            return False, f"Ollama check failed: {e}"

    def list_models(self) -> list[str]:
        """List available Ollama models.

        Returns:
            List of model names
        """
        try:
            url = f"{self.base_url}/api/tags"
            request = urllib.request.Request(url, method="GET")

            with urllib.request.urlopen(request, timeout=10) as response:
                data = loads(response.read())
                return [m["name"] for m in data.get("models", [])]

        except Exception:
            return []

    def generate(self, prompt: str) -> InferenceResult:
        """Generate a response using Ollama.

        Args:
            prompt: The prompt to generate a response for

        Returns:
            InferenceResult with the generated response
        """
        start_time = time.time()

        try:
            messages = self.build_messages(prompt)

            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                    "num_predict": self.config.max_tokens,
                },
            }

            url = f"{self.base_url}/api/chat"
            data = dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(request, timeout=120) as response:
                result = loads(response.read())

            elapsed_ms = (time.time() - start_time) * 1000

            content = result.get("message", {}).get("content", "")
            eval_count = result.get("eval_count", 0)
            eval_duration = result.get("eval_duration", 0)

            # Ollama reports duration in nanoseconds
            if eval_duration > 0:
                tokens_per_sec = eval_count / (eval_duration / 1e9)
            else:
                tokens_per_sec = 0

            return InferenceResult(
                success=True,
                response=content.strip(),
                latency_ms=elapsed_ms,
                tokens_generated=eval_count,
                tokens_per_second=tokens_per_sec,
                metadata={
                    "model": self.model,
                    "backend": self.name,
                },
            )

        except urllib.error.URLError as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return InferenceResult(
                success=False,
                error=f"Cannot connect to Ollama: {e}",
                latency_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return InferenceResult(
                success=False,
                error=str(e),
                latency_ms=elapsed_ms,
            )
