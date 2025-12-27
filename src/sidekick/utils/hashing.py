"""Content hashing utilities for deduplication."""

import hashlib
from pathlib import Path


def hash_content(content: str | bytes) -> str:
    """Generate SHA-256 hash of content.

    Args:
        content: String or bytes to hash

    Returns:
        Hex-encoded SHA-256 hash
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def hash_file(path: Path, chunk_size: int = 8192) -> str:
    """Generate SHA-256 hash of a file.

    Args:
        path: Path to file
        chunk_size: Size of chunks to read

    Returns:
        Hex-encoded SHA-256 hash
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def short_hash(content: str | bytes, length: int = 8) -> str:
    """Generate a short hash for display purposes.

    Args:
        content: String or bytes to hash
        length: Number of characters to return

    Returns:
        Truncated hex-encoded hash
    """
    return hash_content(content)[:length]
