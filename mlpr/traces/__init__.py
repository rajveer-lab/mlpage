"""Page reference traces: synthetic patterns, program-derived traces and real block I/O traces."""

from .io import load_trace_file
from .msr import load_msr
from .programs import bfs_trace, matmul_trace, mergesort_trace
from .synthetic import (
    TEXTBOOK,
    loop_trace,
    mixed_trace,
    phase_trace,
    scan_hot_trace,
    zipf_trace,
)

__all__ = [
    "TEXTBOOK",
    "bfs_trace",
    "load_msr",
    "load_trace_file",
    "loop_trace",
    "matmul_trace",
    "mergesort_trace",
    "mixed_trace",
    "phase_trace",
    "scan_hot_trace",
    "zipf_trace",
]
