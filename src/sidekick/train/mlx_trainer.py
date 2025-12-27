"""MLX training backend for Apple Silicon."""

import subprocess
import sys
import time
from pathlib import Path

from sidekick.core.models import QAPair
from sidekick.core.paths import get_paths
from sidekick.train.base import BaseTrainer, TrainingConfig, TrainingResult
from sidekick.utils.fast_json import dumps


class MLXTrainer(BaseTrainer):
    """Training backend using MLX for Apple Silicon.

    Uses mlx-lm for efficient fine-tuning on Apple Silicon Macs.
    """

    # Model mapping to HuggingFace repos with MLX weights
    MODEL_MAP = {
        # Qwen3 models (Dec 2025) - recommended
        "qwen3-0.6b": "mlx-community/Qwen3-0.6B-4bit",
        "qwen3-4b": "mlx-community/Qwen3-4B-4bit",
        "qwen3-8b": "mlx-community/Qwen3-8B-4bit",
        # Legacy Qwen2.5 models
        "qwen2.5-0.5b": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "qwen2.5-1.5b": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "qwen2.5-3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "phi-3-mini": "mlx-community/Phi-3-mini-4k-instruct-4bit",
    }

    @property
    def name(self) -> str:
        return "mlx"

    def _get_model_path(self) -> str:
        """Get the MLX model path for the configured base model."""
        base = self.config.base_model.lower()

        # Check for direct mapping
        for key, value in self.MODEL_MAP.items():
            if key in base:
                return value

        # Fall back to default (Qwen3-4B)
        return self.MODEL_MAP["qwen3-4b"]

    def train(self, qa_pairs: list[QAPair]) -> TrainingResult:
        """Train using MLX LoRA via CLI.

        Args:
            qa_pairs: QA pairs for training

        Returns:
            TrainingResult
        """
        # Check if mlx_lm is available
        try:
            import mlx_lm  # noqa: F401
        except ImportError:
            return TrainingResult(
                success=False,
                error="mlx-lm not installed. Install with: sidekick setup mlx",
            )

        start_time = time.time()
        paths = get_paths()

        # Prepare output directories
        output_dir = self.config.output_dir or paths.adapters_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        adapter_dir = output_dir / "mlx_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # Prepare training data
        training_data = self.prepare_training_data(qa_pairs)

        # Write training data to JSONL (batch write for efficiency)
        train_file = output_dir / "train.jsonl"
        with open(train_file, "w") as f:
            f.write("\n".join(dumps(ex) for ex in training_data))

        # Create validation split (10%, minimum batch_size examples)
        # For small datasets, use same data for validation
        effective_batch_size = min(self.config.batch_size, len(training_data))
        if len(training_data) < effective_batch_size * 2:
            # Very small dataset - use training data as validation
            valid_data = training_data
        else:
            split_idx = max(effective_batch_size, len(training_data) // 10)
            valid_data = training_data[:split_idx]
        valid_file = output_dir / "valid.jsonl"
        with open(valid_file, "w") as f:
            f.write("\n".join(dumps(ex) for ex in valid_data))

        try:
            model_path = self._get_model_path()
            iters = self.config.epochs * len(training_data)

            # Reduce batch size for small datasets
            effective_batch_size = min(self.config.batch_size, len(training_data))

            # Use CLI for stable API across versions
            cmd = [
                sys.executable, "-m", "mlx_lm", "lora",
                "--train",
                "--model", model_path,
                "--data", str(output_dir),
                "--adapter-path", str(adapter_dir),
                "--iters", str(iters),
                "--batch-size", str(effective_batch_size),
                "--learning-rate", str(self.config.learning_rate),
                "--num-layers", "16",  # Fine-tune 16 layers
            ]

            # Resume from existing adapter for incremental training
            if self.config.resume_from and self.config.resume_from.exists():
                adapter_file = self.config.resume_from / "adapters.safetensors"
                if adapter_file.exists():
                    cmd.extend(["--resume-adapter-file", str(adapter_file)])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                return TrainingResult(
                    success=False,
                    error=f"MLX training failed: {error_msg[:500]}",
                )

            elapsed = time.time() - start_time

            return TrainingResult(
                success=True,
                adapter_path=adapter_dir,
                training_time_seconds=elapsed,
                metrics={
                    "num_examples": len(training_data),
                    "epochs": self.config.epochs,
                    "iters": iters,
                },
            )

        except Exception as e:
            return TrainingResult(
                success=False,
                error=str(e),
            )

    def export_gguf(self, adapter_path: Path, output_path: Path) -> Path:
        """Export MLX model with adapter to GGUF.

        Note: MLX doesn't directly export to GGUF. We need to:
        1. Fuse the adapter with the base model
        2. Convert to safetensors
        3. Use llama.cpp to convert to GGUF

        For now, we'll fuse the adapter and save in MLX format.
        Users can use external tools for GGUF conversion.
        """
        try:
            from mlx_lm import fuse
        except ImportError:
            raise ImportError("mlx-lm required for export")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        model_path = self._get_model_path()

        # Fuse adapter with base model
        fused_dir = output_path.parent / "fused"
        fuse.fuse(
            model=model_path,
            adapter_path=str(adapter_path),
            save_path=str(fused_dir),
        )

        return fused_dir
