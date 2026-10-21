"""Converters from agent-trace formats to trajectory schema v0.3."""

from collections.abc import Iterator
from pathlib import Path

from graduate.core import Session
from graduate.ingest.catalog import CatalogReport, audit
from graduate.ingest.converters.hermes import convert_sharegpt
from graduate.ingest.converters.openai_chat import convert_messages
from graduate.ingest.io import read_records, read_sessions, write_sessions
from graduate.ingest.presets import PRESETS
from graduate.ingest.repair import parse_arguments, strip_template_tokens, tool_call_from_text
from graduate.ingest.validate import IngestReport, report


def ingest(
    path: str | Path, preset: str, source: str | None = None, limit: int | None = None
) -> Iterator[Session]:
    """Convert every record of `path` with a named preset (see `PRESETS`)."""
    convert = PRESETS[preset]
    src = source or f"{preset}:{Path(path).name}"
    for rec in read_records(path, limit):
        yield convert(rec, src)


__all__ = [
    "PRESETS",
    "CatalogReport",
    "IngestReport",
    "audit",
    "convert_messages",
    "convert_sharegpt",
    "ingest",
    "parse_arguments",
    "read_records",
    "read_sessions",
    "report",
    "strip_template_tokens",
    "tool_call_from_text",
    "write_sessions",
]
