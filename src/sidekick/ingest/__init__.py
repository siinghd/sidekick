"""Data ingestion module for Sidekick."""

from sidekick.ingest.auto import AutoIngester, get_ingester
from sidekick.ingest.base import BaseIngester, IngestResult
from sidekick.ingest.csv_ingester import CsvIngester
from sidekick.ingest.html import HtmlIngester
from sidekick.ingest.json_ingester import JsonIngester
from sidekick.ingest.manifest import ManifestManager, get_manifest_manager
from sidekick.ingest.markdown import MarkdownIngester
from sidekick.ingest.mbox import MboxIngester
from sidekick.ingest.pdf import PdfIngester
from sidekick.ingest.text import TextIngester

__all__ = [
    "AutoIngester",
    "BaseIngester",
    "CsvIngester",
    "HtmlIngester",
    "IngestResult",
    "JsonIngester",
    "ManifestManager",
    "MarkdownIngester",
    "MboxIngester",
    "PdfIngester",
    "TextIngester",
    "get_ingester",
    "get_manifest_manager",
]
