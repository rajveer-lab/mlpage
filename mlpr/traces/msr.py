"""Loader for the MSR Cambridge block I/O traces (Narayanan et al., FAST 2008).

The traces are distributed by the SNIA IOTTA repository
(http://iotta.snia.org/traces/block-io/388) under SNIA's trace licence, so
they are not bundled here. Each CSV row is

    Timestamp,Hostname,DiskNumber,Type,Offset,Size,ResponseTime

with Offset and Size in bytes. A request is expanded into every page it
touches, and page numbers are renumbered densely in order of first use.
"""

from __future__ import annotations

import csv
import gzip
from pathlib import Path


def load_msr(
    path: str | Path,
    page_size: int = 4096,
    max_requests: int | None = None,
    max_accesses: int | None = None,
    op: str | None = None,
) -> list[int]:
    """Read an MSR CSV (optionally .gz) as a page reference string.

    ``op`` keeps only "Read" or "Write" requests; by default both are used.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    ids: dict[int, int] = {}
    trace: list[int] = []
    requests = 0
    with opener(path, "rt", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            if op is not None and row[3].strip().lower() != op.lower():
                continue
            try:
                offset, size = int(row[4]), int(row[5])
            except ValueError:  # header line
                continue
            first = offset // page_size
            last = (offset + max(size, 1) - 1) // page_size
            for page in range(first, last + 1):
                trace.append(ids.setdefault(page, len(ids)))
            requests += 1
            if max_requests is not None and requests >= max_requests:
                break
            if max_accesses is not None and len(trace) >= max_accesses:
                del trace[max_accesses:]
                break
    return trace
