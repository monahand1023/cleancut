"""Helpers for hand-editing an EDL (used by `add-cut` and `review` subcommands)."""

from __future__ import annotations

import re


def parse_timestamp(s: str) -> float:
    """Parse "MM:SS", "MM:SS.mmm", "H:MM:SS", "H:MM:SS.mmm", or plain seconds."""
    s = s.strip()
    # Plain seconds
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    parts = s.split(":")
    if len(parts) == 2:
        m, sec = parts
        return int(m) * 60 + float(sec)
    if len(parts) == 3:
        h, m, sec = parts
        return int(h) * 3600 + int(m) * 60 + float(sec)
    raise ValueError(f"Cannot parse timestamp {s!r}; use SS, MM:SS, or H:MM:SS")


def fmt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h}:{m:02d}:{s:05.2f}" if h else f"{m}:{s:05.2f}"


def find_overlapping_shot(seconds: float, shots: list) -> tuple[float, float] | None:
    """Returns (start, end) of the shot containing `seconds`, or None."""
    for s in shots:
        start = s.start if hasattr(s, "start") else s[0]
        end = s.end if hasattr(s, "end") else s[1]
        if start <= seconds < end:
            return (float(start), float(end))
    return None
