"""Prompt-tuned fallback mode - no actual training required.

This mode doesn't fine-tune a model. Instead, it:
1. Compiles QA pairs into a context document
2. Uses a base model with this context injected at inference time

This is useful when:
- No GPU is available for training
- Quick setup without training overhead
- Testing before committing to training

Optimized for:
- Cached context loading (avoids repeated file reads)
- Fast JSON serialization
"""

import time
from pathlib import Path

from sidekick.core.models import QAPair
from sidekick.core.paths import get_paths
from sidekick.train.base import BaseTrainer, TrainingConfig, TrainingResult
from sidekick.utils.fast_json import dumps, loads

# Module-level cache for context (avoids repeated file reads during inference)
_context_cache: dict | None = None
_context_prompt_cache: str | None = None
_cache_mtime: float = 0


class PromptTunedTrainer(BaseTrainer):
    """Prompt-tuned mode - compiles QA into a context document.

    Instead of training, this creates a context file that can be
    injected into prompts at inference time.
    """

    @property
    def name(self) -> str:
        return "prompt_tuned"

    def train(self, qa_pairs: list[QAPair]) -> TrainingResult:
        """Compile QA pairs into a context document.

        Args:
            qa_pairs: QA pairs to compile

        Returns:
            TrainingResult with path to context file
        """
        start_time = time.time()
        paths = get_paths()

        output_dir = self.config.output_dir or paths.merged_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create context document
        context_file = output_dir / "sidekick_context.json"
        context_text_file = output_dir / "sidekick_context.txt"

        # Build structured context
        context = {
            "type": "prompt_tuned",
            "version": "1.0",
            "qa_count": len(qa_pairs),
            "facts": [],
        }

        # Extract facts from QA pairs
        text_lines = ["# Personal Context\n"]
        for qa in qa_pairs:
            context["facts"].append({
                "q": qa.question,
                "a": qa.answer,
            })
            text_lines.append(f"Q: {qa.question}")
            text_lines.append(f"A: {qa.answer}\n")

        # Save JSON format
        with open(context_file, "w") as f:
            f.write(dumps(context, indent=True))

        # Save text format (for injection into prompts)
        with open(context_text_file, "w") as f:
            f.write("\n".join(text_lines))

        elapsed = time.time() - start_time

        return TrainingResult(
            success=True,
            model_path=context_file,
            training_time_seconds=elapsed,
            metrics={
                "num_facts": len(qa_pairs),
                "context_size_bytes": context_file.stat().st_size,
            },
        )

    def export_gguf(self, adapter_path: Path, output_path: Path) -> Path:
        """No GGUF export for prompt-tuned mode.

        In prompt-tuned mode, we use the base model as-is with
        context injection. No model export is needed.

        Returns:
            Path to the context file instead
        """
        paths = get_paths()
        context_file = paths.merged_dir / "sidekick_context.json"

        if context_file.exists():
            return context_file

        raise FileNotFoundError(
            "No context file found. Run 'sidekick train' first."
        )


def load_context() -> dict | None:
    """Load the compiled context for prompt injection.

    Cached: Only reads file if it has changed since last read.

    Returns:
        Context dict or None if not available
    """
    global _context_cache, _cache_mtime

    paths = get_paths()
    context_file = paths.merged_dir / "sidekick_context.json"

    if not context_file.exists():
        return None

    # Check if file has changed
    current_mtime = context_file.stat().st_mtime
    if _context_cache is not None and current_mtime == _cache_mtime:
        return _context_cache

    # Read and cache
    with open(context_file, "rb") as f:
        _context_cache = loads(f.read())
    _cache_mtime = current_mtime

    return _context_cache


def get_context_prompt() -> str | None:
    """Get the context as a prompt prefix.

    Cached: Only reads file if it has changed since last read.

    Returns:
        Context string for prompt injection or None
    """
    global _context_prompt_cache, _cache_mtime

    paths = get_paths()
    context_file = paths.merged_dir / "sidekick_context.txt"

    if not context_file.exists():
        return None

    # Check if file has changed
    current_mtime = context_file.stat().st_mtime
    if _context_prompt_cache is not None and current_mtime == _cache_mtime:
        return _context_prompt_cache

    # Read and cache
    _context_prompt_cache = context_file.read_text()
    _cache_mtime = current_mtime

    return _context_prompt_cache


def inject_context(prompt: str) -> str:
    """Inject personal context into a prompt.

    Args:
        prompt: Original prompt

    Returns:
        Prompt with context injected
    """
    context = get_context_prompt()
    if not context:
        return prompt

    return f"""Here is some personal context about the user:

{context}

Now answer the following question using this context:

{prompt}"""
