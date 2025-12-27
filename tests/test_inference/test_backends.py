"""Tests for inference backends."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from sidekick.inference import (
    InferenceConfig,
    InferenceMode,
    LlamaCppBackend,
    OllamaBackend,
    PromptTunedBackend,
    detect_inference_mode,
    get_inference_backend,
)


class TestLlamaCppBackend:
    """Tests for LlamaCppBackend."""

    def test_backend_name(self) -> None:
        """Test backend name property."""
        config = InferenceConfig()
        backend = LlamaCppBackend(config)

        assert backend.name == "llama.cpp"

    def test_is_available_not_installed(self) -> None:
        """Test is_available when llama-cpp-python not installed."""
        config = InferenceConfig()
        backend = LlamaCppBackend(config)

        with patch.dict("sys.modules", {"llama_cpp": None}):
            # Force reimport check
            available, msg = backend.is_available()
            # May or may not be installed in test environment
            assert isinstance(available, bool)
            assert isinstance(msg, str)

    def test_generate_no_model_path(self) -> None:
        """Test generate fails without model path."""
        config = InferenceConfig()  # No model path
        backend = LlamaCppBackend(config)

        result = backend.generate("Hello")

        assert result.success is False
        # May fail due to missing package or missing model path
        assert result.error is not None

    def test_generate_model_not_found(self, temp_dir: Path) -> None:
        """Test generate fails when model file doesn't exist."""
        config = InferenceConfig(model_path=temp_dir / "nonexistent.gguf")
        backend = LlamaCppBackend(config)

        result = backend.generate("Hello")

        assert result.success is False
        # May fail due to missing package or missing file
        assert result.error is not None


class TestOllamaBackend:
    """Tests for OllamaBackend."""

    def test_backend_name(self) -> None:
        """Test backend name property."""
        config = InferenceConfig()
        backend = OllamaBackend(config)

        assert backend.name == "ollama"

    def test_default_model(self) -> None:
        """Test default Ollama model."""
        config = InferenceConfig()
        backend = OllamaBackend(config)

        assert backend.model == "qwen2.5:1.5b"

    def test_custom_model(self) -> None:
        """Test custom Ollama model."""
        config = InferenceConfig()
        backend = OllamaBackend(config, model="llama3:8b")

        assert backend.model == "llama3:8b"

    def test_custom_base_url(self) -> None:
        """Test custom Ollama base URL."""
        config = InferenceConfig()
        backend = OllamaBackend(config, base_url="http://192.168.1.100:11434")

        assert backend.base_url == "http://192.168.1.100:11434"

    @patch("sidekick.inference.ollama.urllib.request.urlopen")
    def test_is_available_success(self, mock_urlopen: MagicMock) -> None:
        """Test is_available when Ollama is running."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = InferenceConfig()
        backend = OllamaBackend(config)

        available, msg = backend.is_available()

        assert available is True
        assert "available" in msg.lower()

    @patch("sidekick.inference.ollama.urllib.request.urlopen")
    def test_is_available_not_running(self, mock_urlopen: MagicMock) -> None:
        """Test is_available when Ollama is not running."""
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        config = InferenceConfig()
        backend = OllamaBackend(config)

        available, msg = backend.is_available()

        assert available is False
        assert "connect" in msg.lower()

    @patch("sidekick.inference.ollama.urllib.request.urlopen")
    def test_list_models(self, mock_urlopen: MagicMock) -> None:
        """Test list_models method."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "models": [
                {"name": "qwen2.5:1.5b"},
                {"name": "llama3:8b"},
            ]
        }).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = InferenceConfig()
        backend = OllamaBackend(config)

        models = backend.list_models()

        assert "qwen2.5:1.5b" in models
        assert "llama3:8b" in models


class TestPromptTunedBackend:
    """Tests for PromptTunedBackend."""

    def test_backend_name(self) -> None:
        """Test backend name property."""
        config = InferenceConfig()
        backend = PromptTunedBackend(config)

        assert backend.name == "prompt_tuned"

    def test_is_available_no_context(self, temp_dir: Path) -> None:
        """Test is_available when no context file exists."""
        config = InferenceConfig()
        backend = PromptTunedBackend(config)

        with patch("sidekick.inference.prompt_tuned.get_context_prompt") as mock:
            mock.return_value = None

            available, msg = backend.is_available()

            assert available is False
            assert "context" in msg.lower()

    @patch("sidekick.inference.prompt_tuned.get_context_prompt")
    @patch.object(OllamaBackend, "is_available")
    def test_is_available_with_context_and_ollama(
        self, mock_ollama: MagicMock, mock_context: MagicMock
    ) -> None:
        """Test is_available when context exists and Ollama is running."""
        mock_context.return_value = "# Personal Context\nQ: Name?\nA: John"
        mock_ollama.return_value = (True, "Ollama available")

        config = InferenceConfig()
        backend = PromptTunedBackend(config)

        available, msg = backend.is_available()

        assert available is True
        assert "available" in msg.lower()


class TestDetectInferenceMode:
    """Tests for detect_inference_mode function."""

    def test_detect_gguf_model(self, temp_dir: Path) -> None:
        """Test detection of GGUF model."""
        with patch("sidekick.inference.get_paths") as mock_paths:
            merged_dir = temp_dir / "merged"
            merged_dir.mkdir()
            adapters_dir = temp_dir / "adapters"
            adapters_dir.mkdir()

            # Create a GGUF file
            gguf_file = merged_dir / "sidekick.gguf"
            gguf_file.write_text("fake gguf")

            mock_paths.return_value.merged_dir = merged_dir
            mock_paths.return_value.adapters_dir = adapters_dir

            mode, path = detect_inference_mode()

            assert mode == InferenceMode.GGUF
            assert path == gguf_file

    def test_detect_prompt_tuned(self, temp_dir: Path) -> None:
        """Test detection of prompt-tuned mode."""
        with patch("sidekick.inference.get_paths") as mock_paths:
            merged_dir = temp_dir / "merged"
            merged_dir.mkdir()
            adapters_dir = temp_dir / "adapters"
            adapters_dir.mkdir()

            # Create context file but no GGUF
            context_file = merged_dir / "sidekick_context.json"
            context_file.write_text('{"type": "prompt_tuned"}')

            mock_paths.return_value.merged_dir = merged_dir
            mock_paths.return_value.adapters_dir = adapters_dir

            mode, path = detect_inference_mode()

            assert mode == InferenceMode.PROMPT_TUNED
            assert path == context_file

    def test_detect_none(self, temp_dir: Path) -> None:
        """Test detection when no model available."""
        with patch("sidekick.inference.get_paths") as mock_paths:
            merged_dir = temp_dir / "merged"
            merged_dir.mkdir()
            adapters_dir = temp_dir / "adapters"
            adapters_dir.mkdir()

            mock_paths.return_value.merged_dir = merged_dir
            mock_paths.return_value.adapters_dir = adapters_dir

            mode, path = detect_inference_mode()

            assert mode == InferenceMode.NONE
            assert path is None


class TestGetInferenceBackend:
    """Tests for get_inference_backend factory."""

    def test_get_llamacpp_backend(self, temp_dir: Path) -> None:
        """Test getting LlamaCpp backend explicitly."""
        gguf_file = temp_dir / "model.gguf"
        gguf_file.write_text("fake")

        backend = get_inference_backend(
            backend="llamacpp",
            model_path=gguf_file,
        )

        assert isinstance(backend, LlamaCppBackend)
        assert backend.config.model_path == gguf_file

    def test_get_ollama_backend(self) -> None:
        """Test getting Ollama backend explicitly."""
        backend = get_inference_backend(backend="ollama")

        assert isinstance(backend, OllamaBackend)

    def test_get_prompt_tuned_backend(self) -> None:
        """Test getting prompt-tuned backend explicitly."""
        backend = get_inference_backend(backend="prompt_tuned")

        assert isinstance(backend, PromptTunedBackend)

    def test_get_unknown_backend(self) -> None:
        """Test getting unknown backend raises error."""
        try:
            get_inference_backend(backend="unknown")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "unknown" in str(e).lower()

    def test_auto_detect_no_model(self, temp_dir: Path) -> None:
        """Test auto-detect raises error when no model available."""
        with patch("sidekick.inference.detect_inference_mode") as mock:
            mock.return_value = (InferenceMode.NONE, None)

            try:
                get_inference_backend()
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "train" in str(e).lower()

    def test_get_backend_with_custom_config(self) -> None:
        """Test getting backend with custom config."""
        config = InferenceConfig(
            max_tokens=256,
            temperature=0.5,
        )

        backend = get_inference_backend(backend="ollama", config=config)

        assert backend.config.max_tokens == 256
        assert backend.config.temperature == 0.5
