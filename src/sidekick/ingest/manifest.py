"""Manifest tracking for ingested sources."""

import json
from pathlib import Path
from typing import Any

from sidekick.core.models import Manifest, Source
from sidekick.core.paths import get_paths


class ManifestManager:
    """Manages the manifest file for tracking ingested sources.

    The manifest tracks:
    - All ingested sources with their metadata
    - Which sources have been processed for QA generation
    - Statistics about the data
    """

    def __init__(self, manifest_path: Path | None = None) -> None:
        """Initialize the manifest manager.

        Args:
            manifest_path: Path to manifest file (default: ~/.sidekick/data/qa/manifest.json)
        """
        self._path = manifest_path or get_paths().manifest_file
        self._manifest: Manifest | None = None

    @property
    def path(self) -> Path:
        """Path to the manifest file."""
        return self._path

    def load(self) -> Manifest:
        """Load the manifest from disk.

        Returns:
            Manifest instance (creates new if doesn't exist)
        """
        if self._manifest is not None:
            return self._manifest

        if not self._path.exists():
            self._manifest = Manifest()
            return self._manifest

        try:
            data = json.loads(self._path.read_text())
            self._manifest = Manifest.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            # Corrupted or invalid - start fresh
            self._manifest = Manifest()

        return self._manifest

    def save(self) -> None:
        """Save the manifest to disk."""
        if self._manifest is None:
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable dict
        data = self._manifest_to_dict(self._manifest)
        self._path.write_text(json.dumps(data, indent=2, default=str))

    def _manifest_to_dict(self, manifest: Manifest) -> dict[str, Any]:
        """Convert manifest to JSON-serializable dict."""
        data = manifest.model_dump()
        # Convert Path objects to strings
        for source_id, source in data.get("sources", {}).items():
            if source.get("path"):
                source["path"] = str(source["path"])
        return data

    def add_source(self, source: Source, save: bool = True) -> bool:
        """Add a source to the manifest.

        Args:
            source: Source to add
            save: Whether to save immediately

        Returns:
            True if source was new, False if duplicate
        """
        manifest = self.load()

        # Check for duplicate
        if source.id in manifest.sources:
            return False

        manifest.add_source(source)

        if save:
            self.save()

        return True

    def get_source(self, source_id: str) -> Source | None:
        """Get a source by ID.

        Args:
            source_id: Source ID

        Returns:
            Source or None if not found
        """
        manifest = self.load()
        return manifest.sources.get(source_id)

    def get_all_sources(self) -> list[Source]:
        """Get all sources.

        Returns:
            List of all sources
        """
        manifest = self.load()
        return list(manifest.sources.values())

    def get_sources_by_tag(self, tag: str) -> list[Source]:
        """Get sources with a specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of matching sources
        """
        manifest = self.load()
        return [s for s in manifest.sources.values() if tag in s.tags]

    def get_unprocessed_sources(self) -> list[Source]:
        """Get sources that haven't been processed for QA.

        Returns:
            List of unprocessed sources
        """
        manifest = self.load()
        return manifest.get_unprocessed_sources()

    def mark_processed(self, source_id: str, save: bool = True) -> None:
        """Mark a source as processed.

        Args:
            source_id: Source ID
            save: Whether to save immediately
        """
        manifest = self.load()
        manifest.mark_processed(source_id)
        if save:
            self.save()

    def update_qa_count(self, count: int, save: bool = True) -> None:
        """Update the QA pair count.

        Args:
            count: New total count
            save: Whether to save immediately
        """
        manifest = self.load()
        manifest.qa_count = count
        if save:
            self.save()

    def source_exists(self, content_hash: str) -> bool:
        """Check if a source with this content hash exists.

        Args:
            content_hash: Content hash to check

        Returns:
            True if source exists
        """
        manifest = self.load()
        for source in manifest.sources.values():
            if source.content_hash == content_hash:
                return True
        return False

    def remove_source(self, content_hash: str, save: bool = True) -> bool:
        """Remove a source by content hash.

        Args:
            content_hash: Content hash of source to remove
            save: Whether to save immediately

        Returns:
            True if source was removed, False if not found
        """
        manifest = self.load()
        paths = get_paths()

        # Find source by content hash
        source_id = None
        for sid, source in manifest.sources.items():
            if source.content_hash == content_hash:
                source_id = sid
                break

        if source_id is None:
            return False

        # Remove from manifest
        del manifest.sources[source_id]

        # Remove processed file
        processed_file = paths.processed_dir / f"{source_id}.txt"
        if processed_file.exists():
            processed_file.unlink()

        if save:
            self.save()

        return True

    def get_stats(self) -> dict[str, int]:
        """Get statistics about ingested sources.

        Returns:
            Dict with counts
        """
        manifest = self.load()
        sources = list(manifest.sources.values())

        processed = sum(1 for s in sources if s.processed)
        total_bytes = sum(s.size_bytes for s in sources)

        return {
            "total_sources": len(sources),
            "processed": processed,
            "unprocessed": len(sources) - processed,
            "total_bytes": total_bytes,
            "qa_pairs": manifest.qa_count,
        }


# Singleton instance
_manager: ManifestManager | None = None


def get_manifest_manager() -> ManifestManager:
    """Get the global ManifestManager instance."""
    global _manager
    if _manager is None:
        _manager = ManifestManager()
    return _manager
