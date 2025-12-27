"""QA generation module for Sidekick."""

from sidekick.qa.deduper import (
    calculate_similarity,
    deduplicate,
    deduplicate_against_existing,
)
from sidekick.qa.generator import (
    AnthropicBackend,
    GenerationResult,
    LlamaCppBackend,
    OllamaBackend,
    OpenAIBackend,
    QAGenerator,
    QAGeneratorBackend,
    get_generator,
)
from sidekick.qa.merger import QADataset, get_qa_dataset

__all__ = [
    "AnthropicBackend",
    "GenerationResult",
    "LlamaCppBackend",
    "OllamaBackend",
    "OpenAIBackend",
    "QADataset",
    "QAGenerator",
    "QAGeneratorBackend",
    "calculate_similarity",
    "deduplicate",
    "deduplicate_against_existing",
    "get_generator",
    "get_qa_dataset",
]
