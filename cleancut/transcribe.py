"""Whisper-based transcription fallback when no .srt is supplied."""

from __future__ import annotations

from pathlib import Path

from cleancut.subtitles import Subtitle


def transcribe(video_path: Path, model_name: str = "base") -> list[Subtitle]:
    try:
        import whisper  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Whisper not installed. Install with: pip install -e '.[whisper]'"
        ) from e

    model = whisper.load_model(model_name)
    result = model.transcribe(str(video_path), verbose=False, word_timestamps=False)

    subs: list[Subtitle] = []
    for i, seg in enumerate(result.get("segments", []), start=1):
        subs.append(
            Subtitle(
                index=i,
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=str(seg["text"]).strip(),
            )
        )
    return subs
