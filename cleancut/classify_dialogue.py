"""LLM-based contextual dialogue classifier.

A regex wordlist catches the word "fuck" but misses a scene that's clearly
sexual or drug-related from *context* alone. This module groups the Whisper
transcript into scene-sized chunks and asks a local LLM (via Ollama) to
classify each chunk. Chunks the LLM flags as drug/sex/violence become
EDL `cut` decisions.

Runs entirely on-device. Default model: llama3.1:8b (~5GB, ~30 tok/s on M-series).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from tqdm import tqdm

from cleancut.edl import EditDecision, EditDecisionList
from cleancut.subtitles import Subtitle


SYSTEM_PROMPT = """You are a content classifier for a movie editing tool that removes drug, sex, and violent scenes for family viewing.

You will receive a contiguous chunk of movie dialogue with timestamps. Classify the *scene as a whole*, reading context — not just isolated words.

Output a single JSON object exactly matching this schema:
{
  "category": one of: "clean" | "profanity" | "drugs" | "sex" | "violence" | "multi",
  "should_cut": boolean,
  "confidence": number from 0.0 to 1.0,
  "reasoning": string (one short sentence)
}

Rules:
- "drugs"   = drug use, dealing, paraphernalia, kicking habit, getting high
- "sex"     = sexual activity, sexual coercion, intimate dialogue
- "violence"= explicit violent intent or active violence (not action-movie fighting)
- "profanity" alone NEVER triggers should_cut=true. Mute, don't cut.
- should_cut=true ONLY when the scene's plot purpose is the drug/sex/violent content itself
- Be conservative: if uncertain, prefer "clean" or "profanity" with should_cut=false
- A martial-arts fight is not "violence" for these purposes
- An angry character cursing is not "violence"

Output JSON only. No prose, no markdown, no commentary."""


@dataclass
class LLMParams:
    model: str = "llama3.1:8b"
    chunk_max_seconds: float = 90.0      # max length of a scene chunk
    chunk_join_gap: float = 6.0          # gap between subtitle lines that breaks a chunk
    min_chunk_lines: int = 2             # skip chunks with fewer than this many lines
    min_confidence: float = 0.6          # below this, ignore the classification
    pad_seconds: float = 1.0
    ollama_host: str | None = None       # None = default 127.0.0.1:11434


@dataclass
class DialogueChunk:
    start: float
    end: float
    lines: list[Subtitle]

    def format_for_prompt(self) -> str:
        out = []
        for s in self.lines:
            mm = int(s.start // 60)
            ss = s.start - mm * 60
            out.append(f"[{mm}:{ss:05.2f}] {s.text.strip()}")
        return "\n".join(out)


def chunk_dialogue(subs: list[Subtitle], params: LLMParams) -> list[DialogueChunk]:
    """Group consecutive subtitle lines into scene-sized chunks.

    Splits on long silence gaps (> chunk_join_gap) or when a chunk exceeds
    chunk_max_seconds. Drops tiny chunks (fewer than min_chunk_lines).
    """
    chunks: list[DialogueChunk] = []
    current: list[Subtitle] = []
    for s in subs:
        if current:
            gap = s.start - current[-1].end
            span = s.end - current[0].start
            if gap > params.chunk_join_gap or span > params.chunk_max_seconds:
                chunks.append(DialogueChunk(current[0].start, current[-1].end, list(current)))
                current = []
        current.append(s)
    if current:
        chunks.append(DialogueChunk(current[0].start, current[-1].end, list(current)))
    return [c for c in chunks if len(c.lines) >= params.min_chunk_lines]


def _strip_to_json(text: str) -> str:
    """Extract the first JSON object from a possibly chatty LLM response."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else text


def _classify_one(client, params: LLMParams, chunk: DialogueChunk) -> dict | None:
    """Send a single chunk to Ollama; return parsed JSON or None on failure."""
    prompt = (
        f"Scene from {chunk.start:.1f}s to {chunk.end:.1f}s "
        f"(duration {chunk.end - chunk.start:.1f}s):\n\n{chunk.format_for_prompt()}\n\n"
        "Return the JSON classification now."
    )
    try:
        resp = client.chat(
            model=params.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={"temperature": 0.0, "num_ctx": 4096},
        )
        text = resp["message"]["content"]
        return json.loads(_strip_to_json(text))
    except Exception:
        return None


def classify_dialogue(subs: list[Subtitle], params: LLMParams) -> EditDecisionList:
    """Classify every dialogue chunk; emit cut decisions for flagged scenes."""
    try:
        import ollama
    except ImportError as e:
        raise RuntimeError(
            "LLM classifier requires the ollama python client. "
            "Install with: pip install 'cleancut[llm]'"
        ) from e

    chunks = chunk_dialogue(subs, params)
    if not chunks:
        return EditDecisionList()

    client = ollama.Client(host=params.ollama_host) if params.ollama_host else ollama.Client()

    # Warm-load the model so first-chunk latency doesn't break the tqdm ETA.
    try:
        client.generate(model=params.model, prompt="ok", options={"num_predict": 1})
    except Exception:
        pass

    edl = EditDecisionList()
    for chunk in tqdm(chunks, desc="LLM dialogue scan", unit="chunk", leave=False):
        result = _classify_one(client, params, chunk)
        if not result:
            continue
        if not result.get("should_cut"):
            continue
        confidence = float(result.get("confidence", 0.0))
        if confidence < params.min_confidence:
            continue
        category = str(result.get("category", "multi"))
        # Profanity alone should never cut.
        if category == "profanity":
            continue
        if category not in {"drugs", "sex", "violence", "multi"}:
            continue
        edl.add(
            EditDecision(
                start=max(0.0, chunk.start - params.pad_seconds),
                end=chunk.end + params.pad_seconds,
                action="cut",
                category=category if category != "multi" else "sex+drugs",
                reason=f"LLM ({params.model}): {result.get('reasoning', '')[:140]}",
                source="llm-dialogue",
            )
        )

    return edl
