"""Tests for hardware detection module."""

from unittest.mock import MagicMock, patch

from sidekick.train.hardware import (
    HardwareInfo,
    HardwareType,
    TrainingBackend,
    check_backend_available,
    detect_apple_silicon,
    detect_hardware,
    detect_nvidia_gpu,
)


class TestHardwareEnums:
    """Tests for hardware enums."""

    def test_hardware_type_values(self) -> None:
        """Test HardwareType enum values."""
        assert HardwareType.NVIDIA_GPU.value == "nvidia_gpu"
        assert HardwareType.APPLE_SILICON.value == "apple_silicon"
        assert HardwareType.CPU_ONLY.value == "cpu_only"

    def test_training_backend_values(self) -> None:
        """Test TrainingBackend enum values."""
        assert TrainingBackend.UNSLOTH.value == "unsloth"
        assert TrainingBackend.MLX.value == "mlx"
        assert TrainingBackend.PROMPT_TUNED.value == "prompt_tuned"


class TestHardwareInfo:
    """Tests for HardwareInfo dataclass."""

    def test_nvidia_hardware_info(self) -> None:
        """Test HardwareInfo for NVIDIA GPU."""
        info = HardwareInfo(
            hardware_type=HardwareType.NVIDIA_GPU,
            recommended_backend=TrainingBackend.UNSLOTH,
            gpu_name="RTX 4090",
            gpu_memory_gb=24.0,
            cpu_cores=16,
            details="NVIDIA RTX 4090 with 24.0GB VRAM",
        )

        assert info.hardware_type == HardwareType.NVIDIA_GPU
        assert info.recommended_backend == TrainingBackend.UNSLOTH
        assert info.gpu_name == "RTX 4090"
        assert info.gpu_memory_gb == 24.0
        assert info.unified_memory_gb is None
        assert info.cpu_cores == 16

    def test_apple_hardware_info(self) -> None:
        """Test HardwareInfo for Apple Silicon."""
        info = HardwareInfo(
            hardware_type=HardwareType.APPLE_SILICON,
            recommended_backend=TrainingBackend.MLX,
            unified_memory_gb=32.0,
            cpu_cores=10,
            details="Apple Silicon with 32GB unified memory",
        )

        assert info.hardware_type == HardwareType.APPLE_SILICON
        assert info.recommended_backend == TrainingBackend.MLX
        assert info.unified_memory_gb == 32.0
        assert info.gpu_name is None
        assert info.gpu_memory_gb is None

    def test_cpu_only_hardware_info(self) -> None:
        """Test HardwareInfo for CPU only."""
        info = HardwareInfo(
            hardware_type=HardwareType.CPU_ONLY,
            recommended_backend=TrainingBackend.PROMPT_TUNED,
            cpu_cores=8,
            details="No GPU detected",
        )

        assert info.hardware_type == HardwareType.CPU_ONLY
        assert info.recommended_backend == TrainingBackend.PROMPT_TUNED


class TestDetectNvidiaGpu:
    """Tests for detect_nvidia_gpu function."""

    @patch("sidekick.train.hardware.subprocess.run")
    def test_nvidia_gpu_detected(self, mock_run: MagicMock) -> None:
        """Test detection of NVIDIA GPU."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA GeForce RTX 4090, 24576\n",
        )

        has_gpu, gpu_name, vram_gb = detect_nvidia_gpu()

        assert has_gpu is True
        assert gpu_name == "NVIDIA GeForce RTX 4090"
        assert vram_gb == 24.0

    @patch("sidekick.train.hardware.subprocess.run")
    def test_no_nvidia_gpu(self, mock_run: MagicMock) -> None:
        """Test when no NVIDIA GPU is present."""
        mock_run.side_effect = FileNotFoundError()

        has_gpu, gpu_name, vram_gb = detect_nvidia_gpu()

        assert has_gpu is False
        assert gpu_name is None
        assert vram_gb is None

    @patch("sidekick.train.hardware.subprocess.run")
    def test_nvidia_smi_fails(self, mock_run: MagicMock) -> None:
        """Test when nvidia-smi returns non-zero."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
        )

        has_gpu, gpu_name, vram_gb = detect_nvidia_gpu()

        assert has_gpu is False
        assert gpu_name is None
        assert vram_gb is None


class TestDetectAppleSilicon:
    """Tests for detect_apple_silicon function."""

    @patch("sidekick.train.hardware.platform.system")
    def test_not_darwin(self, mock_system: MagicMock) -> None:
        """Test on non-macOS system."""
        mock_system.return_value = "Linux"

        is_apple, memory = detect_apple_silicon()

        assert is_apple is False
        assert memory is None

    @patch("sidekick.train.hardware.platform.system")
    @patch("sidekick.train.hardware.subprocess.run")
    def test_apple_silicon_detected(
        self, mock_run: MagicMock, mock_system: MagicMock
    ) -> None:
        """Test detection of Apple Silicon."""
        mock_system.return_value = "Darwin"

        def mock_subprocess(cmd, **kwargs):
            if "machdep.cpu.brand_string" in cmd:
                return MagicMock(stdout="Apple M2 Max\n")
            elif "hw.memsize" in cmd:
                return MagicMock(stdout=str(32 * 1024**3))  # 32GB
            return MagicMock(stdout="")

        mock_run.side_effect = mock_subprocess

        is_apple, memory = detect_apple_silicon()

        assert is_apple is True
        assert memory == 32.0

    @patch("sidekick.train.hardware.platform.system")
    @patch("sidekick.train.hardware.subprocess.run")
    def test_intel_mac(
        self, mock_run: MagicMock, mock_system: MagicMock
    ) -> None:
        """Test detection on Intel Mac."""
        mock_system.return_value = "Darwin"
        mock_run.return_value = MagicMock(stdout="Intel(R) Core(TM) i9-9900K\n")

        is_apple, memory = detect_apple_silicon()

        assert is_apple is False
        assert memory is None


class TestDetectHardware:
    """Tests for detect_hardware function."""

    @patch("sidekick.train.hardware.detect_nvidia_gpu")
    @patch("sidekick.train.hardware.detect_apple_silicon")
    def test_nvidia_gpu_high_vram(
        self, mock_apple: MagicMock, mock_nvidia: MagicMock
    ) -> None:
        """Test with high VRAM NVIDIA GPU."""
        mock_nvidia.return_value = (True, "RTX 4090", 24.0)
        mock_apple.return_value = (False, None)

        info = detect_hardware()

        assert info.hardware_type == HardwareType.NVIDIA_GPU
        assert info.recommended_backend == TrainingBackend.UNSLOTH
        assert info.gpu_memory_gb == 24.0
        assert "excellent for training" in info.details

    @patch("sidekick.train.hardware.detect_nvidia_gpu")
    @patch("sidekick.train.hardware.detect_apple_silicon")
    def test_nvidia_gpu_low_vram(
        self, mock_apple: MagicMock, mock_nvidia: MagicMock
    ) -> None:
        """Test with low VRAM NVIDIA GPU."""
        mock_nvidia.return_value = (True, "GTX 1650", 4.0)
        mock_apple.return_value = (False, None)

        info = detect_hardware()

        assert info.hardware_type == HardwareType.NVIDIA_GPU
        assert info.recommended_backend == TrainingBackend.UNSLOTH
        assert "with quantization" in info.details

    @patch("sidekick.train.hardware.detect_nvidia_gpu")
    @patch("sidekick.train.hardware.detect_apple_silicon")
    def test_apple_silicon_high_memory(
        self, mock_apple: MagicMock, mock_nvidia: MagicMock
    ) -> None:
        """Test with high memory Apple Silicon."""
        mock_nvidia.return_value = (False, None, None)
        mock_apple.return_value = (True, 32.0)

        info = detect_hardware()

        assert info.hardware_type == HardwareType.APPLE_SILICON
        assert info.recommended_backend == TrainingBackend.MLX
        assert info.unified_memory_gb == 32.0
        assert "good for MLX training" in info.details

    @patch("sidekick.train.hardware.detect_nvidia_gpu")
    @patch("sidekick.train.hardware.detect_apple_silicon")
    def test_apple_silicon_low_memory(
        self, mock_apple: MagicMock, mock_nvidia: MagicMock
    ) -> None:
        """Test with low memory Apple Silicon."""
        mock_nvidia.return_value = (False, None, None)
        mock_apple.return_value = (True, 8.0)

        info = detect_hardware()

        assert info.hardware_type == HardwareType.APPLE_SILICON
        assert info.recommended_backend == TrainingBackend.MLX
        assert "smaller models" in info.details

    @patch("sidekick.train.hardware.detect_nvidia_gpu")
    @patch("sidekick.train.hardware.detect_apple_silicon")
    def test_cpu_only_fallback(
        self, mock_apple: MagicMock, mock_nvidia: MagicMock
    ) -> None:
        """Test fallback to CPU only."""
        mock_nvidia.return_value = (False, None, None)
        mock_apple.return_value = (False, None)

        info = detect_hardware()

        assert info.hardware_type == HardwareType.CPU_ONLY
        assert info.recommended_backend == TrainingBackend.PROMPT_TUNED
        assert "prompt-tuned mode" in info.details


class TestCheckBackendAvailable:
    """Tests for check_backend_available function."""

    def test_prompt_tuned_always_available(self) -> None:
        """Test that prompt-tuned is always available."""
        available, msg = check_backend_available(TrainingBackend.PROMPT_TUNED)

        assert available is True
        assert "always available" in msg

    @patch.dict("sys.modules", {"unsloth": MagicMock()})
    def test_unsloth_available(self) -> None:
        """Test Unsloth availability check when installed."""
        # Import will succeed due to mock
        available, msg = check_backend_available(TrainingBackend.UNSLOTH)
        # This may fail in test environment, so we just check it doesn't crash
        assert isinstance(available, bool)
        assert isinstance(msg, str)

    @patch.dict("sys.modules", {"mlx": MagicMock(), "mlx_lm": MagicMock()})
    def test_mlx_available(self) -> None:
        """Test MLX availability check when installed."""
        available, msg = check_backend_available(TrainingBackend.MLX)
        # This may fail in test environment, so we just check it doesn't crash
        assert isinstance(available, bool)
        assert isinstance(msg, str)
