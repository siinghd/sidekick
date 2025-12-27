"""Hardware detection for training backend selection."""

import platform
import subprocess
from dataclasses import dataclass
from enum import Enum


class HardwareType(str, Enum):
    """Type of hardware available for training."""

    NVIDIA_GPU = "nvidia_gpu"
    APPLE_SILICON = "apple_silicon"
    CPU_ONLY = "cpu_only"


class TrainingBackend(str, Enum):
    """Available training backends."""

    UNSLOTH = "unsloth"  # NVIDIA GPU with Unsloth
    MLX = "mlx"  # Apple Silicon with MLX
    PROMPT_TUNED = "prompt_tuned"  # No training, just context injection


@dataclass
class HardwareInfo:
    """Information about available hardware."""

    hardware_type: HardwareType
    recommended_backend: TrainingBackend
    gpu_name: str | None = None
    gpu_memory_gb: float | None = None
    unified_memory_gb: float | None = None
    cpu_cores: int | None = None
    details: str = ""


def detect_nvidia_gpu() -> tuple[bool, str | None, float | None]:
    """Detect NVIDIA GPU presence and specs.

    Returns:
        Tuple of (has_gpu, gpu_name, vram_gb)
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            if output:
                parts = output.split(",")
                name = parts[0].strip()
                memory_mb = float(parts[1].strip())
                memory_gb = memory_mb / 1024
                return True, name, memory_gb
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    return False, None, None


def detect_apple_silicon() -> tuple[bool, float | None]:
    """Detect Apple Silicon and unified memory.

    Returns:
        Tuple of (is_apple_silicon, unified_memory_gb)
    """
    if platform.system() != "Darwin":
        return False, None

    # Check for Apple Silicon
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        cpu_brand = result.stdout.strip().lower()
        is_apple = "apple" in cpu_brand

        if is_apple:
            # Get total memory
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            memory_bytes = int(result.stdout.strip())
            memory_gb = memory_bytes / (1024 ** 3)
            return True, memory_gb

    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, Exception):
        pass

    return False, None


def detect_hardware() -> HardwareInfo:
    """Detect available hardware and recommend training backend.

    Returns:
        HardwareInfo with detected specs and recommendations
    """
    import os

    cpu_cores = os.cpu_count()

    # Check for NVIDIA GPU first
    has_nvidia, gpu_name, vram_gb = detect_nvidia_gpu()
    if has_nvidia and vram_gb:
        if vram_gb >= 8:
            return HardwareInfo(
                hardware_type=HardwareType.NVIDIA_GPU,
                recommended_backend=TrainingBackend.UNSLOTH,
                gpu_name=gpu_name,
                gpu_memory_gb=vram_gb,
                cpu_cores=cpu_cores,
                details=f"NVIDIA {gpu_name} with {vram_gb:.1f}GB VRAM - excellent for training",
            )
        elif vram_gb >= 4:
            return HardwareInfo(
                hardware_type=HardwareType.NVIDIA_GPU,
                recommended_backend=TrainingBackend.UNSLOTH,
                gpu_name=gpu_name,
                gpu_memory_gb=vram_gb,
                cpu_cores=cpu_cores,
                details=f"NVIDIA {gpu_name} with {vram_gb:.1f}GB VRAM - can train with quantization",
            )

    # Check for Apple Silicon
    is_apple, unified_memory = detect_apple_silicon()
    if is_apple and unified_memory:
        if unified_memory >= 16:
            return HardwareInfo(
                hardware_type=HardwareType.APPLE_SILICON,
                recommended_backend=TrainingBackend.MLX,
                unified_memory_gb=unified_memory,
                cpu_cores=cpu_cores,
                details=f"Apple Silicon with {unified_memory:.0f}GB unified memory - good for MLX training",
            )
        elif unified_memory >= 8:
            return HardwareInfo(
                hardware_type=HardwareType.APPLE_SILICON,
                recommended_backend=TrainingBackend.MLX,
                unified_memory_gb=unified_memory,
                cpu_cores=cpu_cores,
                details=f"Apple Silicon with {unified_memory:.0f}GB - can train smaller models",
            )

    # Fall back to prompt-tuned mode
    return HardwareInfo(
        hardware_type=HardwareType.CPU_ONLY,
        recommended_backend=TrainingBackend.PROMPT_TUNED,
        cpu_cores=cpu_cores,
        details="No GPU detected - using prompt-tuned mode (no training required)",
    )


def check_backend_available(backend: TrainingBackend) -> tuple[bool, str]:
    """Check if a training backend is available.

    Args:
        backend: Backend to check

    Returns:
        Tuple of (available, message)
    """
    if backend == TrainingBackend.UNSLOTH:
        try:
            import unsloth  # noqa: F401

            return True, "Unsloth is available"
        except ImportError:
            return False, "Unsloth not installed. Install with: pip install unsloth"

    elif backend == TrainingBackend.MLX:
        try:
            import mlx  # noqa: F401
            import mlx_lm  # noqa: F401

            return True, "MLX is available"
        except ImportError:
            return False, "MLX not installed. Install with: pip install mlx mlx-lm"

    elif backend == TrainingBackend.PROMPT_TUNED:
        return True, "Prompt-tuned mode is always available"

    return False, f"Unknown backend: {backend}"
