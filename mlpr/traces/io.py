"""Plain-text traces: page numbers separated by whitespace or commas; '#' starts a comment."""

from __future__ import annotations

from pathlib import Path


def load_trace_file(path: str | Path) -> list[int]:
    trace: list[int] = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].replace(",", " ")
        trace.extend(int(tok) for tok in line.split())
    return trace
