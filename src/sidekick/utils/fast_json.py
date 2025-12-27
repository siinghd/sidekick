"""Fast JSON utilities with orjson fallback.

Uses orjson when available (3-10x faster), falls back to stdlib json.
"""

from typing import Any

# Try to import orjson for faster JSON parsing
try:
    import orjson

    def dumps(obj: Any, *, indent: bool = False) -> str:
        """Serialize object to JSON string."""
        opts = orjson.OPT_INDENT_2 if indent else 0
        return orjson.dumps(obj, option=opts).decode("utf-8")

    def dumps_bytes(obj: Any) -> bytes:
        """Serialize object to JSON bytes."""
        return orjson.dumps(obj)

    def loads(s: str | bytes) -> Any:
        """Deserialize JSON string to object."""
        return orjson.loads(s)

    FAST_JSON = True

except ImportError:
    import json

    def dumps(obj: Any, *, indent: bool = False) -> str:
        """Serialize object to JSON string."""
        return json.dumps(obj, indent=2 if indent else None, separators=(",", ":") if not indent else None)

    def dumps_bytes(obj: Any) -> bytes:
        """Serialize object to JSON bytes."""
        return json.dumps(obj, separators=(",", ":")).encode("utf-8")

    def loads(s: str | bytes) -> Any:
        """Deserialize JSON string to object."""
        if isinstance(s, bytes):
            s = s.decode("utf-8")
        return json.loads(s)

    FAST_JSON = False


def load_file(path) -> Any:
    """Load JSON from file path."""
    with open(path, "rb") as f:
        return loads(f.read())


def save_file(path, obj: Any, *, indent: bool = True) -> None:
    """Save object to JSON file."""
    with open(path, "w") as f:
        f.write(dumps(obj, indent=indent))
