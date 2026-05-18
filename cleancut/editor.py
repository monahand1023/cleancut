"""ffmpeg orchestration: apply cuts, mutes, and burned subtitles."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from cleancut.edl import EditDecisionList
from cleancut.subtitles import Subtitle


def _require_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg (e.g. brew install ffmpeg).")
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe not found on PATH. It ships with ffmpeg.")


def probe_duration(path: Path) -> float:
    _require_ffmpeg()
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", str(path),
        ]
    )
    return float(json.loads(out)["format"]["duration"])


@dataclass
class Range:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def keep_segments(duration: float, cuts: list[Range]) -> list[Range]:
    """Complement of cuts within [0, duration]. Returns the segments we keep."""
    cuts = sorted(cuts, key=lambda r: r.start)
    kept: list[Range] = []
    cursor = 0.0
    for c in cuts:
        s = max(c.start, 0.0)
        e = min(c.end, duration)
        if e <= cursor:
            continue
        if s > cursor:
            kept.append(Range(cursor, s))
        cursor = max(cursor, e)
    if cursor < duration:
        kept.append(Range(cursor, duration))
    return [r for r in kept if r.duration > 0.001]


def shift_after_cuts(t: float, cuts: list[Range]) -> float | None:
    """Map a source-timeline timestamp to the cut-output timeline.

    Returns None if `t` falls inside a removed segment.
    """
    out = t
    for c in sorted(cuts, key=lambda r: r.start):
        if c.start >= t:
            break
        if c.start <= t <= c.end:
            return None
        out -= c.duration
    return max(0.0, out)


def adjust_subtitles_for_cuts(
    subs: list[Subtitle], cuts: list[Range]
) -> list[Subtitle]:
    """Shift / trim / drop subtitles to match a video with the given cuts removed."""
    if not cuts:
        return list(subs)
    cuts = sorted(cuts, key=lambda r: r.start)
    out: list[Subtitle] = []
    next_idx = 1
    for s in subs:
        new_start = shift_after_cuts(s.start, cuts)
        new_end = shift_after_cuts(s.end, cuts)
        # Both endpoints fall inside cuts -> drop.
        if new_start is None and new_end is None:
            continue
        # Start cut out: snap to the next keep boundary.
        if new_start is None:
            for c in cuts:
                if c.start <= s.start <= c.end:
                    snapped = shift_after_cuts(c.end + 1e-4, cuts) or 0.0
                    new_start = snapped
                    break
        if new_end is None:
            for c in cuts:
                if c.start <= s.end <= c.end:
                    snapped = shift_after_cuts(c.start - 1e-4, cuts)
                    new_end = snapped if snapped is not None else new_start
                    break
        if new_start is None or new_end is None or new_end <= new_start:
            continue
        out.append(Subtitle(index=next_idx, start=new_start, end=new_end, text=s.text))
        next_idx += 1
    return out


def shift_ranges_after_cuts(ranges: list[Range], cuts: list[Range]) -> list[Range]:
    """Map mute ranges from source timeline to cut-output timeline."""
    out: list[Range] = []
    for r in ranges:
        ns = shift_after_cuts(r.start, cuts)
        ne = shift_after_cuts(r.end, cuts)
        if ns is None and ne is None:
            continue
        if ns is None:
            ns = 0.0
        if ne is None:
            # Range tail falls inside a cut: trim to the cut boundary.
            for c in sorted(cuts, key=lambda x: x.start):
                if c.start <= r.end <= c.end:
                    snapped = shift_after_cuts(c.start - 1e-4, cuts)
                    if snapped is not None:
                        ne = snapped
                    break
        if ne is None or ne <= ns:
            continue
        out.append(Range(ns, ne))
    return out


def _video_encoder_args(encoder: str, quality: int) -> list[str]:
    """Return ffmpeg flags for the chosen video encoder."""
    if encoder == "videotoolbox":
        # videotoolbox uses -q:v (higher = better). Map CRF-ish quality to a sensible q.
        # CRF 18 ~ q 65, CRF 20 ~ q 60, CRF 23 ~ q 50.
        q = max(30, min(100, 90 - quality * 2))
        return ["-c:v", "h264_videotoolbox", "-q:v", str(q), "-b:v", "0"]
    # libx264 default
    return ["-c:v", "libx264", "-preset", "slow", "-crf", str(quality)]


def apply_cuts(
    input_path: Path,
    cuts: list[Range],
    output_path: Path,
    encoder: str = "libx264",
    quality: int = 20,
) -> None:
    """Re-encode `input_path` with `cuts` removed, writing to `output_path`."""
    _require_ffmpeg()
    if not cuts:
        # Nothing to cut — just remux.
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(input_path), "-c", "copy", str(output_path)],
            check=True,
        )
        return

    duration = probe_duration(input_path)
    segments = keep_segments(duration, cuts)
    if not segments:
        raise RuntimeError("All segments cut — nothing left to render.")

    parts: list[str] = []
    concat_inputs: list[str] = []
    for i, seg in enumerate(segments):
        parts.append(
            f"[0:v]trim=start={seg.start:.3f}:end={seg.end:.3f},"
            f"setpts=PTS-STARTPTS[v{i}];"
            f"[0:a]atrim=start={seg.start:.3f}:end={seg.end:.3f},"
            f"asetpts=PTS-STARTPTS[a{i}]"
        )
        concat_inputs.append(f"[v{i}][a{i}]")
    filter_complex = ";".join(parts) + ";" + "".join(concat_inputs) + (
        f"concat=n={len(segments)}:v=1:a=1[outv][outa]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        *_video_encoder_args(encoder, quality),
        "-c:a", "aac", "-b:a", "192k",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def _ffmpeg_has_libass() -> bool:
    """Check if the installed ffmpeg can run the subtitles= filter (needs libass)."""
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-filters"], stderr=subprocess.STDOUT, text=True
        )
        return any(line.split()[1:2] == ["subtitles"] for line in out.splitlines() if line.strip())
    except Exception:
        return False


def apply_mutes_and_subs(
    input_path: Path,
    mutes: list[Range],
    srt_path: Path | None,
    output_path: Path,
    burn_subs: bool = True,
    encoder: str = "libx264",
    quality: int = 20,
) -> None:
    """Apply mute ranges via volume filter; add subtitles either as burn-in (libass)
    or as a soft subtitle track in the container (always works).

    Soft subs are the default unless `burn_subs=True` AND ffmpeg has libass.
    Soft subs are faster (video can be stream-copied) and toggleable in players.
    """
    _require_ffmpeg()

    can_burn = burn_subs and srt_path and srt_path.exists() and _ffmpeg_has_libass()

    cmd: list[str] = ["ffmpeg", "-y", "-i", str(input_path)]

    # If soft-subs mode, add the SRT as a second input.
    has_soft_subs = srt_path and srt_path.exists() and not can_burn
    if has_soft_subs:
        cmd += ["-i", str(srt_path)]

    # Audio filter: mute volumes in the given ranges.
    if mutes:
        enable = "+".join(f"between(t,{r.start:.3f},{r.end:.3f})" for r in mutes)
        cmd += ["-af", f"volume=enable='{enable}':volume=0"]

    safe_dir: Path | None = None
    if can_burn:
        import shutil as _sh
        safe_dir = Path("/tmp/cleancut-render")
        safe_dir.mkdir(parents=True, exist_ok=True)
        safe_srt = safe_dir / "subs.srt"
        _sh.copy(str(srt_path), str(safe_srt))
        cmd += ["-vf", "subtitles=subs.srt"]
        cmd += _video_encoder_args(encoder, quality)
    elif has_soft_subs:
        # Stream-copy video, encode subs as mov_text into the MP4 container.
        cmd += ["-map", "0:v", "-map", "0:a", "-map", "1:0"]
        cmd += ["-c:v", "copy"]
        cmd += ["-c:s", "mov_text"]
        cmd += ["-metadata:s:s:0", "language=eng",
                "-metadata:s:s:0", "title=cleancut (softened)"]
    else:
        cmd += ["-c:v", "copy"]

    cmd += ["-c:a", "aac", "-b:a", "192k", str(output_path)]
    cwd = str(safe_dir) if can_burn else None
    subprocess.run(cmd, check=True, cwd=cwd)


def edl_to_ranges(edl: EditDecisionList, action: str) -> list[Range]:
    return [Range(d.start, d.end) for d in edl.by_action(action)]
