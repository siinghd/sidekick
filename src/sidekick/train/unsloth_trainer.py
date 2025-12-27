"""Unsloth training backend for NVIDIA GPUs."""

import time
from pathlib import Path

from sidekick.core.models import QAPair
from sidekick.core.paths import get_paths
from sidekick.train.base import BaseTrainer, TrainingConfig, TrainingResult


class UnslothTrainer(BaseTrainer):
    """Training backend using Unsloth for NVIDIA GPUs.

    Unsloth provides 2-5x faster training with 80% less memory.
    """

    # Model mapping to Unsloth-compatible models
    MODEL_MAP = {
        # Qwen3 models (Dec 2025) - recommended
        "qwen3-0.6b": "unsloth/Qwen3-0.6B-bnb-4bit",
        "qwen3-4b": "unsloth/Qwen3-4B-bnb-4bit",
        "qwen3-8b": "unsloth/Qwen3-8B-bnb-4bit",
        # Legacy Qwen2.5 models
        "qwen2.5-0.5b": "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        "qwen2.5-1.5b": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        "qwen2.5-3b": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "phi-3-mini": "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
    }

    @property
    def name(self) -> str:
        return "unsloth"

    def _get_model_name(self) -> str:
        """Get the Unsloth model name for the configured base model."""
        base = self.config.base_model.lower()

        for key, value in self.MODEL_MAP.items():
            if key in base:
                return value

        # Fall back to default (Qwen3-4B)
        return self.MODEL_MAP["qwen3-4b"]

    def train(self, qa_pairs: list[QAPair]) -> TrainingResult:
        """Train using Unsloth LoRA.

        Args:
            qa_pairs: QA pairs for training

        Returns:
            TrainingResult
        """
        try:
            from unsloth import FastLanguageModel
            from trl import SFTTrainer
            from transformers import TrainingArguments
            from datasets import Dataset
        except ImportError as e:
            return TrainingResult(
                success=False,
                error=f"Required packages not installed: {e}. "
                "Install with: pip install unsloth transformers trl datasets",
            )

        start_time = time.time()
        paths = get_paths()

        output_dir = self.config.output_dir or paths.adapters_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        adapter_dir = output_dir / "unsloth_adapter"

        try:
            model_name = self._get_model_name()

            # Load model with Unsloth
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=self.config.max_seq_length,
                load_in_4bit=True,
            )

            # Add LoRA adapters
            model = FastLanguageModel.get_peft_model(
                model,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
            )

            # Prepare dataset
            training_data = self.prepare_training_data(qa_pairs)

            def format_prompt(example):
                messages = example["messages"]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                return {"text": text}

            dataset = Dataset.from_list(training_data)
            dataset = dataset.map(format_prompt)

            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(adapter_dir),
                num_train_epochs=self.config.epochs,
                per_device_train_batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                warmup_ratio=0.1,
                logging_steps=10,
                save_strategy="epoch",
                fp16=True,
            )

            # Train
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=self.config.max_seq_length,
                args=training_args,
            )

            trainer.train()
            trainer.save_model(str(adapter_dir))

            elapsed = time.time() - start_time

            return TrainingResult(
                success=True,
                adapter_path=adapter_dir,
                training_time_seconds=elapsed,
                metrics={
                    "num_examples": len(training_data),
                    "epochs": self.config.epochs,
                },
            )

        except Exception as e:
            return TrainingResult(
                success=False,
                error=str(e),
            )

    def export_gguf(self, adapter_path: Path, output_path: Path) -> Path:
        """Export model with adapter to GGUF format.

        Args:
            adapter_path: Path to LoRA adapter
            output_path: Output GGUF file path

        Returns:
            Path to GGUF file
        """
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError("Unsloth required for export")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        model_name = self._get_model_name()

        # Load model and adapter
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=True,
        )

        # Load adapter
        model.load_adapter(str(adapter_path))

        # Merge and save
        model = model.merge_and_unload()

        # Save to GGUF
        gguf_path = output_path.with_suffix(".gguf")
        model.save_pretrained_gguf(
            str(gguf_path.parent),
            tokenizer,
            quantization_method=self.config.quantization,
        )

        return gguf_path
