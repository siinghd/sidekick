"""Pydantic models for Sidekick data structures."""

from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Type of source document."""

    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    PDF = "pdf"
    MBOX = "mbox"
    HTML = "html"
    CSV = "csv"
    UNKNOWN = "unknown"


class Source(BaseModel):
    """Represents a source document that has been ingested."""

    id: str = Field(description="Unique identifier (content hash)")
    path: Path | None = Field(default=None, description="Original file path if from file")
    source_type: SourceType = Field(description="Type of source")
    content_hash: str = Field(description="SHA-256 hash of content")
    size_bytes: int = Field(description="Size in bytes")
    added_at: datetime = Field(default_factory=datetime.now, description="When added")
    tags: list[str] = Field(default_factory=list, description="User-assigned tags")
    processed: bool = Field(default=False, description="Whether QA has been generated")


class QAPair(BaseModel):
    """A question-answer pair for training."""

    question: str = Field(description="The question")
    answer: str = Field(description="The answer")
    source_id: str = Field(description="ID of source document")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    created_at: datetime = Field(default_factory=datetime.now)


class Manifest(BaseModel):
    """Tracks what has been processed."""

    sources: dict[str, Source] = Field(default_factory=dict, description="All ingested sources")
    qa_count: int = Field(default=0, description="Total QA pairs generated")
    last_updated: datetime = Field(default_factory=datetime.now)
    version: str = Field(default="1", description="Manifest schema version")

    def add_source(self, source: Source) -> None:
        """Add or update a source."""
        self.sources[source.id] = source
        self.last_updated = datetime.now()

    def get_unprocessed_sources(self) -> list[Source]:
        """Get sources that haven't been processed for QA generation."""
        return [s for s in self.sources.values() if not s.processed]

    def mark_processed(self, source_id: str) -> None:
        """Mark a source as processed."""
        if source_id in self.sources:
            self.sources[source_id].processed = True
            self.last_updated = datetime.now()


class TrainingMessage(BaseModel):
    """A single message in the training format."""

    role: str = Field(description="Message role (user/assistant)")
    content: str = Field(description="Message content")


class TrainingExample(BaseModel):
    """A training example in the messages format."""

    messages: list[TrainingMessage] = Field(description="List of messages")

    @classmethod
    def from_qa(cls, qa: QAPair) -> "TrainingExample":
        """Create a training example from a QA pair."""
        return cls(
            messages=[
                TrainingMessage(role="user", content=qa.question),
                TrainingMessage(role="assistant", content=qa.answer),
            ]
        )


class ModelInfo(BaseModel):
    """Information about a trained model."""

    name: str = Field(description="Model name")
    base_model: str = Field(description="Base model used")
    trained_at: datetime = Field(default_factory=datetime.now)
    qa_pairs_count: int = Field(description="Number of QA pairs trained on")
    sources_count: int = Field(description="Number of sources used")
    path: Path = Field(description="Path to model file")
    quantization: str | None = Field(default=None, description="Quantization used")
    size_bytes: int = Field(default=0, description="Model file size")
