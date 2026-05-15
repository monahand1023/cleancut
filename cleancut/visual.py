"""Visual scene detection using NudeNet on sampled frames."""

from __future__ import annotations

from pathlib import Path

from cleancut.config import Config
from cleancut.edl import EditDecision, EditDecisionList

# NudeNet class names that signal explicit content.
EXPLICIT_CLASSES = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}

# Borderline classes — flag but don't auto-cut at lower confidence.
SUGGESTIVE_CLASSES = {
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
    "FEMALE_GENITALIA_COVERED",
    "FEET_EXPOSED",
    "BELLY_EXPOSED",
}


def scan_video(video_path: Path, config: Config) -> EditDecisionList:
    """Sample frames from the video and run NudeNet on each.

    Returns an EDL with one EditDecision per flagged frame. Adjacent frames
    are merged later in the pipeline.
    """
    try:
        import cv2  # type: ignore
        from nudenet import NudeDetector  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Visual detection requires extras. Install with: pip install -e '.[visual]'"
        ) from e

    detector = NudeDetector()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps > 0 else 0.0

    sample_interval = max(1, int(round(fps * config.visual_sample_seconds)))
    edl = EditDecisionList()

    frame_idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        t = frame_idx / fps

        # NudeNet detect() accepts file paths or numpy arrays.
        try:
            detections = detector.detect(frame)
        except Exception:
            # Some NudeNet builds require a path. Write a temp jpg.
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, frame)
                detections = detector.detect(tmp.name)

        explicit_hits = [
            d for d in detections
            if d.get("class") in EXPLICIT_CLASSES
            and float(d.get("score", 0)) >= config.visual_threshold
        ]
        if explicit_hits:
            classes = ", ".join(sorted(set(d["class"] for d in explicit_hits)))
            action = config.actions.get("nudity", "cut")
            edl.add(
                EditDecision(
                    start=t,
                    end=t + config.visual_sample_seconds,
                    action=action,
                    category="nudity",
                    reason=f"visual: {classes}",
                    source="visual",
                )
            )

        frame_idx += sample_interval
        if duration and t >= duration:
            break

    cap.release()
    return edl
