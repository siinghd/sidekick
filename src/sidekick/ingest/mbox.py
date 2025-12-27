"""Mbox email file ingester."""

import email
import email.policy
import mailbox
from pathlib import Path
from typing import Iterator

from sidekick.core.models import Source, SourceType
from sidekick.ingest.base import BaseIngester, IngestResult
from sidekick.utils.hashing import hash_file


class MboxIngester(BaseIngester):
    """Ingester for mbox email archive files.

    Extracts text from email messages, including subject,
    from, to, and body content.
    """

    @property
    def source_type(self) -> SourceType:
        return SourceType.MBOX

    @property
    def supported_extensions(self) -> list[str]:
        return [".mbox", ".mbx"]

    def ingest_file(self, path: Path, tags: list[str] | None = None) -> IngestResult:
        """Ingest an mbox file.

        Args:
            path: Path to the mbox file
            tags: Optional tags

        Returns:
            IngestResult with source and extracted text
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content_hash = hash_file(path)
        messages: list[str] = []
        message_count = 0

        try:
            mbox = mailbox.mbox(path)
            for msg in mbox:
                message_count += 1
                text = self._extract_message(msg)
                if text:
                    messages.append(text)
        except Exception as e:
            raise ValueError(f"Failed to parse mbox file: {e}")

        text = "\n\n---\n\n".join(messages)

        source = Source(
            id=content_hash[:16],
            path=path.absolute(),
            source_type=self.source_type,
            content_hash=content_hash,
            size_bytes=path.stat().st_size,
            tags=tags or [],
        )

        return IngestResult(
            source=source,
            text=text,
            metadata={"message_count": message_count},
        )

    def ingest_text(self, text: str, tags: list[str] | None = None) -> IngestResult:
        """Mbox ingester doesn't support raw text input.

        Raises:
            NotImplementedError: Always
        """
        raise NotImplementedError("Mbox ingester requires a file path")

    def _extract_message(self, msg: mailbox.mboxMessage) -> str:
        """Extract text from a single email message.

        Args:
            msg: Email message

        Returns:
            Formatted text representation
        """
        parts: list[str] = []

        # Headers
        subject = msg.get("Subject", "(no subject)")
        from_addr = msg.get("From", "")
        to_addr = msg.get("To", "")
        date = msg.get("Date", "")

        parts.append(f"Subject: {subject}")
        if from_addr:
            parts.append(f"From: {from_addr}")
        if to_addr:
            parts.append(f"To: {to_addr}")
        if date:
            parts.append(f"Date: {date}")

        parts.append("")  # Blank line

        # Body
        body = self._get_body(msg)
        if body:
            parts.append(body)

        return "\n".join(parts)

    def _get_body(self, msg: mailbox.mboxMessage) -> str:
        """Extract the body text from an email.

        Args:
            msg: Email message

        Returns:
            Body text (plain text preferred)
        """
        body_parts: list[str] = []

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                # Skip attachments
                if "attachment" in content_disposition:
                    continue

                # Prefer plain text
                if content_type == "text/plain":
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or "utf-8"
                            text = payload.decode(charset, errors="ignore")
                            body_parts.append(text)
                    except Exception:
                        continue

        else:
            content_type = msg.get_content_type()
            if content_type == "text/plain":
                try:
                    payload = msg.get_payload(decode=True)
                    if payload:
                        charset = msg.get_content_charset() or "utf-8"
                        text = payload.decode(charset, errors="ignore")
                        body_parts.append(text)
                except Exception:
                    pass

        return "\n".join(body_parts).strip()
