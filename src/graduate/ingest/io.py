"""Read source records (JSON, JSONL, Parquet) and read/write sessions as JSONL."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from graduate.core import Session


def read_records(path: str | Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    path = Path(path)
    n = 0
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq  # optional dependency: tf-ingest[parquet]

        pf = pq.ParquetFile(path)
        for g in range(pf.num_row_groups):
            for rec in pf.read_row_group(g).to_pylist():
                if limit is not None and n >= limit:
                    return
                n += 1
                yield rec
        return
    if path.suffix == ".jsonl":
        with path.open() as f:
            for lineno, line in enumerate(f, 1):
                if limit is not None and n >= limit:
                    return
                if line.strip():
                    n += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as e:  # name the file's line, not the record's
                        msg = f"line {lineno}: {e.msg}"
                        raise json.JSONDecodeError(msg, e.doc, e.pos) from None
                    yield record
        return
    data = json.loads(path.read_text())
    records = data.get("simulations", [data]) if isinstance(data, dict) else data
    yield from records[:limit]


def write_sessions(sessions: Iterable[Session], path: str | Path) -> int:
    n = 0
    with Path(path).open("w") as f:
        for s in sessions:
            f.write(s.model_dump_json(exclude_defaults=True) + "\n")
            n += 1
    return n


def read_sessions(path: str | Path) -> Iterator[Session]:
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield Session.model_validate_json(line)
