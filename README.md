# cleancut

Auto-edit movies for content. Detects profanity, drug references, sex, and nudity using a layered "belt and suspenders" stack of local signals, then mutes audio, cuts scenes, and rewrites subtitles to softer text. Outputs a cleaned `.mp4` with a toggleable softened subtitle track.

All AI runs **on-device** — Whisper for transcription, Ollama-hosted LLMs for context, NudeNet + LLaVA for vision, HuggingFace AST for audio events. No cloud APIs.

## Signal stack

| # | Signal | Catches | Cost | Tool |
|---|---|---|---|---|
| 1 | **Wordlist match** | Explicit profanity / drugs / sex words | free | regex |
| 2 | **Context gating** | Drops ambiguous weak hits (e.g. "blow" alone) without strong neighbors | free | regex |
| 3 | **Density clustering** | Clusters of any wordlist hits → likely a "content scene" | free | python |
| 4 | **LLM dialogue** | Contextual scenes wordlists miss ("as good as your ex" + drug pushing) | free local | Ollama + Llama 3.1 8B |
| 5 | **NudeNet** | Explicit nudity (narrow technical kind) | free local | NudeNet |
| 6 | **VLM scene** | Intimate framing without explicit nudity, drug paraphernalia | free local | Ollama + LLaVA 7B |
| 7 | **Audio events** | Moans, screams, gunshots — semantic sound events | free local | HuggingFace AST |

The signals run independently and their results merge into one EDL (Edit Decision List). Density and snap-to-shot post-processing dedupe and align them. The output report shows *which* signal contributed to each cut, so it's auditable.

## Install

Requires Python 3.10+ and `ffmpeg` on PATH.

```bash
brew install ffmpeg ollama
ollama serve &                         # starts the Ollama daemon
ollama pull llama3.1:8b                # for LLM dialogue (~5GB)
ollama pull llava:7b                   # for VLM visual (~5GB)

cd ~/Development/cleancut
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[full]"               # whisper + nudenet + scenedetect + ollama + transformers
```

Lighter installs:

```bash
pip install -e .                       # subtitle-only (no STT, no visual, no LLM)
pip install -e ".[whisper]"            # + transcription fallback
pip install -e ".[visual,scenes]"      # + NudeNet + PySceneDetect
pip install -e ".[llm]"                # + Ollama python client
```

## Quickstart

```bash
# Step 1: see what's in the file and what cleancut would do
cleancut inspect /path/to/movie.mp4

# Step 2: scan — produces an EDL + report, doesn't render
cleancut scan /path/to/movie.mp4 \
  --preset thorough \
  --save-transcript /path/to/movie.whisper.srt \
  -o /path/to/movie.edl.json

# Step 3: review interactively — approve/reject/trim each cut
cleancut review /path/to/movie.edl.json --subs /path/to/movie.whisper.srt

# Step 4: render
cleancut clean /path/to/movie.mp4 \
  --edl /path/to/movie.edl.json \
  --subs /path/to/movie.whisper.srt \
  -o /path/to/movie.clean.mp4

# Bonus: add a cut by hand if review missed something
cleancut add-cut /path/to/movie.edl.json --start 58:32 --end 1:00:08 \
  --category sex --reason "drug dealer / intimate scene"
```

## Subcommands

### `cleancut inspect FILE`

Shows audio tracks, subtitle tracks (text vs image), sidecar `.srt` files found, and a "plan" of what `clean` would do given current flags.

### `cleancut scan FILE`

Runs the full signal stack and writes:
- `<file>.edl.json` — every decision the detectors made
- `<file>.edl.report.txt` — human-readable report grouped by category
- Optional `--save-transcript FILE` — caches the Whisper transcript so future scans skip the 30-min Whisper pass via `--subs`

### `cleancut review EDL`

Interactive walkthrough of every accepted cut. For each one, extracts a representative frame, shows the surrounding dialogue, and prompts:

- `y` — keep
- `n` — reject (sets `accepted=false`)
- `t START END` — trim the cut range (use `MM:SS` or `H:MM:SS`)
- `o` — open the extracted frame in your default viewer
- `s` — skip without changing
- `q` — save and quit

By default, pure-violence cuts are auto-hidden (fight scenes kept). Pass `--include-violence` to review those too.

### `cleancut add-cut EDL --start MM:SS --end MM:SS`

Manually insert a cut into an EDL. `--snap` extends the range outward to nearest PySceneDetect shot boundaries.

### `cleancut clean FILE -o OUTPUT`

Apply an EDL (either freshly scanned or `--edl FILE`) and render the cleaned video. Subtitles become a soft track in the MP4 container (toggle in any player). With ffmpeg compiled `--enable-libass`, subtitles burn in instead.

## Presets

```bash
cleancut scan FILE --preset {fast|balanced|thorough}
```

| Preset      | Whisper       | Visual sample | Density | LLM | VLM | Audio | Notes                  |
|-------------|---------------|---------------|---------|-----|-----|-------|------------------------|
| `fast`      | base          | 2.0s          | off     | off | off | off   | Quick first pass       |
| `balanced`  | small + words | 1.0s          | on      | off | off | off   | Reasonable default     |
| `thorough`  | large-v3+words| 0.5s          | on      | on  | on  | on    | **Default.** Best quality |

`thorough` is the default. On an M-series Mac with 32GB+ RAM, leave it.

## Configuration

Wordlists and replacements live in `cleancut/data/`. Patterns can be flat strings or `{"pattern": "...", "strength": "strong|weak"}`. **Weak** patterns only fire if a strong hit (any category) is within ±30s — eliminates false positives like "blow" matching "blow to the head" in a boxing scene.

```bash
# Override with your own files
cleancut clean FILE --wordlists my_words.json --replacements my_replacements.json
```

## Categories and actions

| Category   | Default action | Notes                                                          |
|------------|----------------|----------------------------------------------------------------|
| profanity  | mute + soften  | Audio muted, subtitle text softened                            |
| drugs      | mute + soften  | Same                                                           |
| sex        | mute + soften  | Visual matches also trigger `cut`                              |
| violence   | keep           | Off by default — fight scenes preserved                        |
| nudity     | cut            | Visual-only (NudeNet, VLM)                                     |

Override per-category: `--action profanity=cut`, `--enable-category violence`, etc.

## Caching

All heavy computations cache to `~/.cache/cleancut/` keyed by video + config:

- **Whisper transcripts** — pass `--save-transcript FILE` once, then `--subs FILE` on re-runs
- **PySceneDetect shot boundaries** — auto-cached, ~5-10 min savings per re-run
- **NudeNet visual scan results** — auto-cached, ~10-30 min savings
- **Audio event detection** — auto-cached, ~5-10 min savings

Invalidated automatically when the video file changes (mtime/size) or relevant config changes.

## Legal

The Family Movie Act of 2005 (US) permits private home performance of filtered authorized copies of motion pictures. cleancut is a tool for personal use on media you own. Don't redistribute the cleaned output.

## License

MIT
