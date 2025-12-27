"""Tests for prompt-tuned trainer module."""

import json
from pathlib import Path
from unittest.mock import patch

from sidekick.core.models import QAPair
from sidekick.train.base import TrainingConfig
from sidekick.train.prompt_tuned import (
    PromptTunedTrainer,
    get_context_prompt,
    inject_context,
    load_context,
)


class TestPromptTunedTrainer:
    """Tests for PromptTunedTrainer."""

    def test_trainer_name(self) -> None:
        """Test trainer name property."""
        config = TrainingConfig()
        trainer = PromptTunedTrainer(config)

        assert trainer.name == "prompt_tuned"

    def test_train_creates_context_files(self, temp_dir: Path) -> None:
        """Test that train creates context files."""
        config = TrainingConfig(output_dir=temp_dir)
        trainer = PromptTunedTrainer(config)

        qa_pairs = [
            QAPair(
                question="What is your name?",
                answer="My name is John.",
                source_id="test1",
            ),
            QAPair(
                question="Where do you work?",
                answer="I work at Acme Corp.",
                source_id="test2",
            ),
        ]

        result = trainer.train(qa_pairs)

        assert result.success is True
        assert result.model_path is not None
        assert result.model_path.exists()

        # Check JSON file
        context_json = temp_dir / "sidekick_context.json"
        assert context_json.exists()

        with open(context_json) as f:
            context = json.load(f)

        assert context["type"] == "prompt_tuned"
        assert context["version"] == "1.0"
        assert context["qa_count"] == 2
        assert len(context["facts"]) == 2
        assert context["facts"][0]["q"] == "What is your name?"
        assert context["facts"][0]["a"] == "My name is John."

        # Check text file
        context_txt = temp_dir / "sidekick_context.txt"
        assert context_txt.exists()

        text_content = context_txt.read_text()
        assert "# Personal Context" in text_content
        assert "Q: What is your name?" in text_content
        assert "A: My name is John." in text_content

    def test_train_empty_qa_pairs(self, temp_dir: Path) -> None:
        """Test train with empty QA pairs."""
        config = TrainingConfig(output_dir=temp_dir)
        trainer = PromptTunedTrainer(config)

        result = trainer.train([])

        assert result.success is True
        assert result.metrics["num_facts"] == 0

    def test_train_records_metrics(self, temp_dir: Path) -> None:
        """Test that training records metrics."""
        config = TrainingConfig(output_dir=temp_dir)
        trainer = PromptTunedTrainer(config)

        qa_pairs = [
            QAPair(question="Q1?", answer="A1", source_id="test1"),
            QAPair(question="Q2?", answer="A2", source_id="test2"),
            QAPair(question="Q3?", answer="A3", source_id="test3"),
        ]

        result = trainer.train(qa_pairs)

        assert result.success is True
        assert result.metrics["num_facts"] == 3
        assert result.metrics["context_size_bytes"] > 0
        assert result.training_time_seconds >= 0

    def test_export_gguf_returns_context_file(self, temp_dir: Path) -> None:
        """Test that export_gguf returns context file for prompt-tuned."""
        config = TrainingConfig(output_dir=temp_dir)
        trainer = PromptTunedTrainer(config)

        # First, train to create context file
        qa_pairs = [QAPair(question="Q?", answer="A", source_id="test")]
        trainer.train(qa_pairs)

        # Mock get_paths to return our temp_dir
        with patch("sidekick.train.prompt_tuned.get_paths") as mock_paths:
            mock_paths.return_value.merged_dir = temp_dir

            # Now export should return the context file
            result = trainer.export_gguf(temp_dir, temp_dir / "output.gguf")

            assert result == temp_dir / "sidekick_context.json"

    def test_export_gguf_raises_if_no_context(self, temp_dir: Path) -> None:
        """Test that export_gguf raises if no context file exists."""
        config = TrainingConfig(output_dir=temp_dir)
        trainer = PromptTunedTrainer(config)

        with patch("sidekick.train.prompt_tuned.get_paths") as mock_paths:
            mock_paths.return_value.merged_dir = temp_dir / "nonexistent"

            try:
                trainer.export_gguf(temp_dir, temp_dir / "output.gguf")
                assert False, "Should have raised FileNotFoundError"
            except FileNotFoundError as e:
                assert "No context file found" in str(e)


class TestLoadContext:
    """Tests for load_context function."""

    def test_load_context_returns_none_if_missing(self, temp_dir: Path) -> None:
        """Test load_context returns None if context file missing."""
        with patch("sidekick.train.prompt_tuned.get_paths") as mock_paths:
            mock_paths.return_value.merged_dir = temp_dir

            result = load_context()

            assert result is None

    def test_load_context_returns_data(self, temp_dir: Path) -> None:
        """Test load_context returns context data."""
        context_file = temp_dir / "sidekick_context.json"
        context_data = {
            "type": "prompt_tuned",
            "version": "1.0",
            "qa_count": 2,
            "facts": [
                {"q": "Q1?", "a": "A1"},
                {"q": "Q2?", "a": "A2"},
            ],
        }
        with open(context_file, "w") as f:
            json.dump(context_data, f)

        with patch("sidekick.train.prompt_tuned.get_paths") as mock_paths:
            mock_paths.return_value.merged_dir = temp_dir

            result = load_context()

            assert result is not None
            assert result["type"] == "prompt_tuned"
            assert result["qa_count"] == 2
            assert len(result["facts"]) == 2


class TestGetContextPrompt:
    """Tests for get_context_prompt function."""

    def test_get_context_prompt_returns_none_if_missing(
        self, temp_dir: Path
    ) -> None:
        """Test get_context_prompt returns None if file missing."""
        with patch("sidekick.train.prompt_tuned.get_paths") as mock_paths:
            mock_paths.return_value.merged_dir = temp_dir

            result = get_context_prompt()

            assert result is None

    def test_get_context_prompt_returns_text(self, temp_dir: Path) -> None:
        """Test get_context_prompt returns text content."""
        context_file = temp_dir / "sidekick_context.txt"
        content = "# Personal Context\n\nQ: What is your name?\nA: John\n"
        context_file.write_text(content)

        with patch("sidekick.train.prompt_tuned.get_paths") as mock_paths:
            mock_paths.return_value.merged_dir = temp_dir

            result = get_context_prompt()

            assert result == content


class TestInjectContext:
    """Tests for inject_context function."""

    def test_inject_context_with_no_context(self, temp_dir: Path) -> None:
        """Test inject_context returns original prompt if no context."""
        with patch("sidekick.train.prompt_tuned.get_context_prompt") as mock:
            mock.return_value = None

            result = inject_context("What is 2+2?")

            assert result == "What is 2+2?"

    def test_inject_context_with_context(self) -> None:
        """Test inject_context injects context properly."""
        context = "# Personal Context\n\nQ: Name?\nA: John\n"

        with patch(
            "sidekick.train.prompt_tuned.get_context_prompt"
        ) as mock:
            mock.return_value = context

            result = inject_context("What is my name?")

            assert "personal context about the user" in result.lower()
            assert context in result
            assert "What is my name?" in result
