# Sidekick

> **Beta** - Currently tested on macOS (Apple Silicon) only.

**Your AI's AI** - A personal memory model for LLMs.

A tiny, locally-trained model that knows YOU. Train it on your notes, chats, and documents. Any LLM can query it to get personalized context. No databases, no embeddings, no retrieval latency - just a small model that answers questions about you.

## Why Sidekick?

| Traditional RAG | Sidekick |
|-----------------|----------|
| Retrieval latency on every query | Zero retrieval - model IS the memory |
| Embedding drift over time | Stable trained knowledge |
| Database maintenance | Single model file |
| Complex infrastructure | Just a CLI |

## Features

- **Train on anything** - Notes, PDFs, emails, chats, Markdown, JSON, CSV, HTML
- **Works everywhere** - Apple Silicon (MLX), NVIDIA (Unsloth), or CPU (prompt-tuned)
- **Zero retrieval** - Model IS the memory, not a database
- **Incremental learning** - Add new data and resume training
- **LLM-powered ingestion** - Automatically detects relevant fields in any JSON format
- **Universal** - HTTP API, MCP server, or OpenAI-compatible proxy

## Requirements

- Python 3.10+
- macOS with Apple Silicon (M1/M2/M3) for MLX training
- Or NVIDIA GPU for Unsloth training
- Or any system for prompt-tuned mode (no GPU required)

## Installation

```bash
pip install sidekick
```

Or with uv:

```bash
uv add sidekick
```

### Optional Dependencies

```bash
# For local inference (recommended)
pip install sidekick[inference]

# For Apple Silicon training
pip install sidekick[mlx]

# For NVIDIA GPU training
pip install sidekick[train]

# Everything
pip install sidekick[all]
```

## Quick Start

```bash
# 1. Initialize
sidekick init

# 2. Add your data
sidekick add ./my-notes/
sidekick add ~/Documents/resume.pdf
sidekick add "I work at Acme Corp as a software engineer"

# 3. Train (auto-detects your hardware)
sidekick train

# 4. Query
sidekick ask "What do I work on?"

# 5. Start server (optional)
sidekick serve
```

## Commands

### `sidekick init`

Initialize Sidekick. Creates `~/.sidekick/` directory structure.

```bash
sidekick init                    # Default setup
sidekick init --model qwen3-8b  # Use larger model
sidekick init --force            # Reinitialize
```

### `sidekick add`

Add files, directories, or raw text.

```bash
# Add files and directories
sidekick add ./notes/
sidekick add document.pdf
sidekick add ~/Documents/

# Add raw text
sidekick add "My favorite color is blue"

# Add with tags
sidekick add ./work/ --tag work
sidekick add ./personal/ --tag personal

# From stdin
cat notes.txt | sidekick add --stdin

# Force re-ingest existing file
sidekick add ./notes.json --force
```

**Supported formats:** `.txt`, `.md`, `.json`, `.jsonl`, `.pdf`, `.csv`, `.html`, `.mbox`

### `sidekick list`

List ingested sources.

```bash
sidekick list              # All sources
sidekick list --tag work   # Filter by tag
sidekick list --processed  # Only processed sources
```

### `sidekick train`

Train the model on ingested data.

```bash
sidekick train                    # Auto-detect hardware and train
sidekick train --mode prompt      # Prompt-tuned mode (no GPU needed)
sidekick train --backend local    # Use local model for QA generation
sidekick train --backend openai   # Use OpenAI for QA generation
sidekick train --epochs 5         # More training epochs
sidekick train --full             # Reprocess all sources
sidekick train --resume           # Continue from existing adapter
sidekick train --generate-only    # Only generate QA pairs, don't train
sidekick train --skip-generate    # Train on existing QA pairs
```

**Training modes:**
- **MLX** - Apple Silicon Macs (fast, efficient)
- **Unsloth** - NVIDIA GPUs (2-5x faster than standard)
- **Prompt-tuned** - No GPU needed (uses context injection)

### `sidekick ask`

Query your trained model.

```bash
sidekick ask "What projects am I working on?"
sidekick ask "Who is my photographer friend?"
sidekick ask "What's my tech stack?" --verbose
sidekick ask "Summarize my career" --max-tokens 1000
sidekick ask "What do I like?" --json
sidekick ask "Tell me about myself" --temperature 0.9
```

### `sidekick serve`

Start an API server.

```bash
# HTTP API server
sidekick serve
sidekick serve --port 8080
sidekick serve --host 0.0.0.0

# MCP server
sidekick serve --mcp

# OpenAI-compatible proxy (injects your context)
sidekick serve --proxy
```

**HTTP Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/ask?q=...` | GET | Ask a question |
| `/ask` | POST | Ask (JSON body) |
| `/v1/chat/completions` | POST | OpenAI-compatible |
| `/v1/models` | GET | List models |

### `sidekick watch`

Watch files and learn incrementally.

```bash
sidekick watch                      # Watch current directory
sidekick watch ~/notes ~/docs       # Watch multiple directories
sidekick watch -p "*.md,*.org"      # Custom file patterns
sidekick watch --verbose            # Show detailed status
```

### `sidekick config`

View or modify configuration.

```bash
sidekick config              # Show current config
sidekick config --edit       # Open in editor
sidekick config --path       # Show config file path
```

### `sidekick export`

Export model or data.

```bash
sidekick export model ./my-model.gguf  # Export GGUF model
sidekick export qa ./qa-pairs.jsonl    # Export QA dataset
sidekick export context ./context.txt  # Export context file
```

## Configuration

Configuration is stored in `~/.sidekick/config.toml`:

```toml
[model]
base = "qwen3-4b"  # Options: qwen3-0.6b (tiny), qwen3-4b (default), qwen3-8b (best)
quantization = "q4_k_m"

[training]
epochs = 3
batch_size = 4
learning_rate = 0.0002
lora_rank = 16

[server]
host = "127.0.0.1"
port = 8765

[watch]
patterns = ["*.md", "*.txt", "*.json"]
ignore = ["node_modules", ".git", "venv"]
debounce_ms = 2000
rate_limit_per_minute = 30

[hot_memory]
max_entries = 100
max_bytes = 100000
max_age_hours = 24.0
```

## Architecture

```
~/.sidekick/
├── config.toml          # Configuration
├── data/
│   ├── processed/       # Extracted text
│   └── qa/
│       ├── manifest.json  # Source tracking
│       └── dataset.jsonl  # Generated QA pairs
└── models/
    ├── base/            # Downloaded base models
    ├── adapters/        # LoRA adapters
    └── merged/          # Final merged models
```

## How It Works

1. **Ingest**: Add your documents. Sidekick extracts text and detects relevant content fields (uses LLM for JSON/JSONL).

2. **Generate QA**: Sidekick uses an LLM to generate question-answer pairs from your content. These capture facts about you in a format ideal for training.

3. **Train**: Fine-tune a small model (Qwen3-4B by default) on your QA pairs using LoRA. This takes minutes on Apple Silicon.

4. **Query**: The trained model answers questions about you. No retrieval needed - the knowledge is baked into the weights.

## Development

```bash
# Clone and install
git clone https://github.com/siinghd/sidekick
cd sidekick
uv sync --dev

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=sidekick

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/
```

## Performance Optimizations

Sidekick is optimized for efficiency:

- **`__slots__`** on all data classes (~40% memory reduction)
- **orjson** for JSON parsing (3-10x faster, with stdlib fallback)
- **Pre-compiled regex** patterns
- **Cached context loading** (file read once, then O(1))
- **Chunked processing** for large files
- **Automatic batch size adjustment** for small datasets

## Roadmap

- [ ] Linux testing
- [ ] Windows support
- [ ] More training backends
- [ ] Model merging/export improvements
- [ ] Web UI

## License

MIT
