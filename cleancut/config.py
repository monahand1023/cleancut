from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Literal

Category = Literal["profanity", "drugs", "sex", "violence", "nudity"]
Action = Literal["mute", "cut", "keep"]

DEFAULT_ACTIONS: dict[str, Action] = {
    "profanity": "mute",
    "drugs": "mute",
    "sex": "mute",
    "violence": "keep",
    "nudity": "cut",
}


@dataclass
class Config:
    wordlists: dict[str, list[str]] = field(default_factory=dict)
    replacements: dict[str, str] = field(default_factory=dict)
    actions: dict[str, Action] = field(default_factory=lambda: dict(DEFAULT_ACTIONS))
    enabled_categories: set[str] = field(
        default_factory=lambda: {"profanity", "drugs", "sex", "nudity"}
    )
    # Visual sampling: examine 1 frame every N seconds.
    visual_sample_seconds: float = 1.0
    # NudeNet confidence threshold for explicit-class detections.
    visual_threshold: float = 0.55
    # Pad mute/cut ranges by this many seconds on each side so cuts feel natural.
    pad_seconds: float = 0.15
    # Merge adjacent ranges closer than this.
    merge_gap_seconds: float = 0.5
    # Whisper model name (tiny/base/small/medium/large).
    whisper_model: str = "base"

    @classmethod
    def load_defaults(cls) -> Config:
        return cls(
            wordlists=_load_packaged_json("wordlists.json"),
            replacements=_load_packaged_json("replacements.json"),
        )

    def override_wordlists(self, path: Path | None) -> None:
        if path:
            self.wordlists = json.loads(Path(path).read_text())

    def override_replacements(self, path: Path | None) -> None:
        if path:
            self.replacements = json.loads(Path(path).read_text())


def _load_packaged_json(name: str) -> dict:
    pkg = resources.files("cleancut.data")
    return json.loads(pkg.joinpath(name).read_text())
