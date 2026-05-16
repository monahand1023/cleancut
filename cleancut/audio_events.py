"""Audio event detection using the AST (Audio Spectrogram Transformer) model.

Closes the gap on scenes that have:
- No relevant dialogue (so wordlist/LLM miss them)
- No explicit nudity (so NudeNet misses)
- Visual content the VLM may also miss (dark frames, off-camera action)

Examples: a sex scene with moans but covered bodies, a torture scene with
screams off-frame, a shootout audible from another room.

Runs locally via HuggingFace transformers. Model is ~360MB, ~0.5-1s per
10-sec clip on M-series. Cached results invalidated on video/config change.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cleancut.edl import EditDecision, EditDecisionList
from cleancut.scenes import Shot


# Map of cleancut category → list of AudioSet labels (as emitted by the model).
DEFAULT_CATEGORY_LABELS: dict[str, list[str]] = {
    "sex": [
        "Moaning",
        "Sigh",
        "Heavy breathing",
        "Pant",
    ],
    "violence": [
        "Gunshot, gunfire",
        "Machine gun",
        "Fusillade",
        "Cap gun",
        "Artillery fire",
        "Explosion",
        "Screaming",
        "Crying, sobbing",
        "Wail, moan",
    ],
}


@dataclass
class AudioEventParams:
    model: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    # Confidence threshold for accepting an event label.
    threshold: float = 0.45
    # Min shot duration to bother classifying (sec).
    min_shot_duration: float = 1.0
    # Audio clip length (sec) per shot — AST is trained at 10s.
    clip_seconds: float = 8.0
    # Categories to emit cuts for — wired through DEFAULT_CATEGORY_LABELS by key.
    enabled_categories: tuple[str, ...] = ("sex", "violence")
    # Drop the violence category by default (per Dan's preference); flip off via config.
    skip_violence: bool = True
    # Use cache.
    use_cache: bool = True


def _extract_full_audio_to_wav(video: Path, audio_track_index: int | None) -> Path:
    """Pull mono 16kHz wav for AST. If audio_track_index is None, takes the first track."""
    import shutil
    import subprocess
    import tempfile
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH.")
    out = Path(tempfile.mkstemp(prefix="cleancut_ae_", suffix=".wav")[1])
    map_arg = ["-map", f"0:{audio_track_index}"] if audio_track_index is not None else []
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(video),
            *map_arg,
            "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le",
            str(out),
        ],
        check=True,
    )
    return out


def _load_audio(wav_path: Path):
    import librosa  # type: ignore
    audio, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    return audio, sr


def _slice_audio(audio: np.ndarray, sr: int, center_seconds: float, clip_seconds: float) -> np.ndarray:
    """Return a clip_seconds slice centered on `center_seconds`, zero-padded if needed."""
    half = clip_seconds / 2.0
    start = max(0.0, center_seconds - half)
    end = start + clip_seconds
    s_idx = int(round(start * sr))
    e_idx = int(round(end * sr))
    e_idx = min(e_idx, len(audio))
    clip = audio[s_idx:e_idx]
    needed = int(round(clip_seconds * sr))
    if len(clip) < needed:
        pad = np.zeros(needed - len(clip), dtype=clip.dtype)
        clip = np.concatenate([clip, pad])
    return clip


def _load_model(model_name: str):
    try:
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Audio event detection requires extras. "
            "Install with: pip install transformers librosa"
        ) from e

    extractor = ASTFeatureExtractor.from_pretrained(model_name)
    model = ASTForAudioClassification.from_pretrained(model_name)
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device).eval()
    return extractor, model, device


def _classify_clip(extractor, model, device, clip: np.ndarray) -> list[tuple[str, float]]:
    import torch
    inputs = extractor(clip, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    # Sigmoid over multi-label outputs (each class independent).
    probs = torch.sigmoid(logits)[0].cpu().numpy()
    id2label = model.config.id2label
    return [(id2label[i], float(probs[i])) for i in range(len(probs))]


def scan_audio_events(
    video: Path,
    shots: list[Shot],
    params: AudioEventParams,
    audio_track_index: int | None = None,
) -> EditDecisionList:
    """For each shot, classify a centered audio clip and emit cuts for matches."""
    from cleancut import cache as _cache

    shot_fp = None
    if shots:
        shot_fp = {"n": len(shots), "first": (shots[0].start, shots[0].end),
                   "last": (shots[-1].start, shots[-1].end)}
    cats_for_hash = [c for c in params.enabled_categories
                     if not (params.skip_violence and c == "violence")]
    h = _cache.config_hash(
        model=params.model,
        threshold=params.threshold,
        clip_seconds=params.clip_seconds,
        min_shot_duration=params.min_shot_duration,
        categories=sorted(cats_for_hash),
        shots=shot_fp,
        audio_track=audio_track_index,
    )
    if params.use_cache:
        hit = _cache.load(video, "audio_events", h)
        if hit:
            return EditDecisionList(
                decisions=[EditDecision(**d) for d in hit.get("decisions", [])]
            )

    wav = _extract_full_audio_to_wav(video, audio_track_index)
    try:
        audio, sr = _load_audio(wav)
        extractor, model, device = _load_model(params.model)
        edl = EditDecisionList()

        active_categories = [
            c for c in params.enabled_categories
            if not (params.skip_violence and c == "violence")
        ]
        label_to_category: dict[str, str] = {}
        for cat in active_categories:
            for label in DEFAULT_CATEGORY_LABELS.get(cat, []):
                label_to_category[label] = cat
        target_labels = set(label_to_category.keys())

        for shot in tqdm(shots, desc="Audio events", unit="shot", leave=False):
            if shot.duration < params.min_shot_duration:
                continue
            center = shot.start + shot.duration / 2.0
            clip = _slice_audio(audio, sr, center, params.clip_seconds)
            results = _classify_clip(extractor, model, device, clip)
            hits = [(lbl, score) for lbl, score in results
                    if lbl in target_labels and score >= params.threshold]
            if not hits:
                continue
            hits.sort(key=lambda x: -x[1])
            cats = sorted({label_to_category[lbl] for lbl, _ in hits})
            top = hits[0]
            edl.add(
                EditDecision(
                    start=shot.start,
                    end=shot.end,
                    action="cut",
                    category="+".join(cats),
                    reason=f"audio event: {top[0]} ({top[1]:.2f})",
                    source="audio",
                )
            )
    finally:
        wav.unlink(missing_ok=True)

    if params.use_cache:
        _cache.save(video, "audio_events", h, {
            "decisions": [asdict(d) for d in edl.decisions],
        })
    return edl
