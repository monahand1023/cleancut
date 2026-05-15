from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import srt

from cleancut.config import Config
from cleancut.edl import EditDecision, EditDecisionList


@dataclass
class Subtitle:
    index: int
    start: float            # seconds
    end: float              # seconds
    text: str

    @classmethod
    def from_srt(cls, s: srt.Subtitle) -> Subtitle:
        return cls(
            index=s.index,
            start=s.start.total_seconds(),
            end=s.end.total_seconds(),
            text=s.content,
        )


def read_srt(path: Path) -> list[Subtitle]:
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    return [Subtitle.from_srt(s) for s in srt.parse(raw)]


def write_srt(subs: list[Subtitle], path: Path) -> None:
    import datetime as dt

    items = [
        srt.Subtitle(
            index=s.index,
            start=dt.timedelta(seconds=s.start),
            end=dt.timedelta(seconds=s.end),
            content=s.text,
        )
        for s in subs
    ]
    Path(path).write_text(srt.compose(items), encoding="utf-8")


def _compile_patterns(wordlists: dict[str, list[str]]) -> dict[str, list[re.Pattern]]:
    return {
        cat: [re.compile(pat, re.IGNORECASE) for pat in pats]
        for cat, pats in wordlists.items()
    }


def scan_subtitles(subs: list[Subtitle], config: Config) -> EditDecisionList:
    """Match wordlists against subtitle text. Each match -> one EditDecision."""
    patterns = _compile_patterns(config.wordlists)
    edl = EditDecisionList()

    for sub in subs:
        matches: list[tuple[str, str]] = []  # (category, matched_text)
        for category, pats in patterns.items():
            if category not in config.enabled_categories:
                continue
            for pat in pats:
                for m in pat.finditer(sub.text):
                    matches.append((category, m.group(0)))

        if not matches:
            continue

        # Pick the highest-severity category if multiple hit the same line.
        severity = {"profanity": 1, "violence": 2, "drugs": 3, "sex": 4, "nudity": 5}
        category = sorted(matches, key=lambda x: -severity.get(x[0], 0))[0][0]
        action = config.actions.get(category, "mute")
        if action == "keep":
            continue

        text_after = soften_text(sub.text, config.replacements)
        edl.add(
            EditDecision(
                start=sub.start,
                end=sub.end,
                action=action,
                category=category,
                reason=f"matched: {', '.join(sorted(set(m[1].lower() for m in matches)))}",
                text_before=sub.text,
                text_after=text_after,
                source="subtitle",
            )
        )

    return edl


def soften_text(text: str, replacements: dict[str, str]) -> str:
    """Case-preserving word-boundary substitution for each entry in replacements."""
    if not replacements:
        return text
    # Sort longer phrases first so multi-word entries take precedence.
    keys = sorted(replacements.keys(), key=len, reverse=True)
    out = text
    for key in keys:
        repl = replacements[key]
        pattern = re.compile(r"\b" + re.escape(key) + r"\b", re.IGNORECASE)
        out = pattern.sub(lambda m, r=repl: _match_case(m.group(0), r), out)
    return out


def _match_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def softened_subtitles(subs: list[Subtitle], config: Config) -> list[Subtitle]:
    """Return a copy of subs with every line softened by the replacements map."""
    return [
        Subtitle(index=s.index, start=s.start, end=s.end, text=soften_text(s.text, config.replacements))
        for s in subs
    ]


def scan_words(words, config: Config) -> EditDecisionList:
    """Match wordlists against individual word-level Whisper output.

    Produces tight mute ranges around just the offending word(s), instead of
    the entire subtitle line. Adjacent flagged words within the same phrase
    will be merged later in the pipeline.

    `words` is a list of cleancut.transcribe.Word; imported lazily to avoid
    a hard dep on the whisper extras.
    """
    patterns = _compile_patterns(config.wordlists)
    severity = {"profanity": 1, "violence": 2, "drugs": 3, "sex": 4, "nudity": 5}
    edl = EditDecisionList()

    # Build a running list of (start, end, text) windows for multi-word matches.
    # We approximate multi-word phrases by looking at 1-, 2-, and 3-word windows
    # so patterns like "blow job" or "get high" can match.
    n = len(words)
    for i, w in enumerate(words):
        # Build candidate windows starting at i.
        for window_size in (1, 2, 3):
            j = i + window_size
            if j > n:
                break
            window = words[i:j]
            text = " ".join(x.text for x in window).strip(" .,!?-")
            if not text:
                continue
            best: tuple[str, str] | None = None
            best_sev = -1
            for category, pats in patterns.items():
                if category not in config.enabled_categories:
                    continue
                for pat in pats:
                    m = pat.search(text)
                    if m:
                        sev = severity.get(category, 0)
                        if sev > best_sev:
                            best = (category, m.group(0))
                            best_sev = sev
            if best is None:
                continue
            category, matched = best
            action = config.actions.get(category, "mute")
            if action == "keep":
                continue
            edl.add(
                EditDecision(
                    start=window[0].start,
                    end=window[-1].end,
                    action=action,
                    category=category,
                    reason=f"matched: {matched.lower()}",
                    text_before=text,
                    text_after=soften_text(text, config.replacements),
                    source="whisper-word",
                )
            )
            # Don't try shorter windows once we matched; longer phrases take precedence.
            break

    return edl
