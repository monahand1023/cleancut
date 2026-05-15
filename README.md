# cleancut

Auto-edit movies for content. Detects profanity, drug references, and sex/nudity, then mutes audio, cuts scenes, and rewrites subtitles to softer text. Outputs a cleaned video file.

Pipeline:

1. **Subtitles**: parses `.srt` if present, otherwise transcribes with Whisper (defaults to `large-v3`, word-level timestamps, MPS on Apple Silicon).
2. **Dialogue scan**: matches against configurable wordlists (profanity / drugs / sex / violence). When Whisper words are available, mute ranges are word-precise, not line-precise.
3. **Shot boundary detection**: PySceneDetect finds every shot cut.
4. **Visual scan**: samples frames within each shot, runs NudeNet, marks a shot `cut` only if a meaningful fraction of its frames are flagged (kills single-frame false positives).
5. **Snap to shots**: dialogue-driven cuts extend outward to the enclosing shot boundary, so cuts land on real edits — never mid-shot.
6. **Render**: ffmpeg applies the EDL (VideoToolbox hardware encoder on macOS by default) and burns softened subtitles back in.

The EDL is a plain JSON file — you can hand-edit it between `scan` and `clean` to accept/reject individual cuts.

### Presets

| Preset      | Whisper      | Sample rate | Streak | Snap-to-shot | Notes                    |
|-------------|--------------|-------------|--------|--------------|--------------------------|
| `fast`      | base         | 2.0s        | 2      | off          | Quick first pass         |
| `balanced`  | small + words| 1.0s        | 2      | on           | Reasonable default       |
| `thorough`  | large-v3 + words | 0.5s    | 3      | on           | **Default.** Best quality |

`thorough` is the default. On an M-series Mac with 32GB+ RAM, leave it.

## Install

Requires Python 3.10+ and `ffmpeg` on PATH.

```bash
brew install ffmpeg            # macOS
cd ~/Development/cleancut
python -m venv .venv && source .venv/bin/activate
pip install -e ".[full]"       # full = whisper + nudenet
```

Lighter installs:

```bash
pip install -e .                 # subtitle-only (no STT, no visual)
pip install -e ".[whisper]"      # + transcription fallback
pip install -e ".[visual]"       # + nudenet visual scan
pip install -e ".[scenes]"       # + PySceneDetect shot boundaries
pip install -e ".[full]"         # everything (recommended)
```

## Usage

### One-shot auto-clean

```bash
cleancut clean movie.mp4 --subs movie.srt -o movie.clean.mp4
```

Without `--subs`, Whisper will transcribe (slow). Add `--no-visual` to skip the NudeNet pass.

### Scan only (produces an EDL you can edit)

```bash
cleancut scan movie.mp4 --subs movie.srt -o movie.edl.json
```

### Hand-edit and apply

Open the `.edl.json` in any text editor. Each decision has an `accepted` flag and an `action` (`mute` / `cut` / `keep`) you can change. Then:

### Apply an existing EDL

```bash
cleancut clean movie.mp4 --edl movie.edl.json -o movie.clean.mp4
```

### Robustness knobs

```bash
# Force the most thorough Whisper model, with explicit MPS device:
cleancut clean movie.mp4 --whisper-model large-v3 --whisper-device mps

# Sample 2 frames per second visually, require 3 consecutive hits:
cleancut clean movie.mp4 --visual-sample-seconds 0.5 --visual-min-streak 3

# Tighter shot detection (more cuts):
cleancut clean movie.mp4 --scene-threshold 22

# Highest-quality software encode (slow):
cleancut clean movie.mp4 --encoder libx264 --quality 16
```

## Configuration

Wordlists and softening replacements live in `cleancut/data/`:

- `wordlists.json` — categorized regex patterns (`profanity`, `drugs`, `sex`, `violence`)
- `replacements.json` — `"original" -> "softer"` text substitutions for subtitle rewrite

Override with your own file:

```bash
cleancut clean movie.mp4 --wordlists my_words.json --replacements my_replacements.json
```

## Categories and actions

| Category   | Default action     | Notes                                                                  |
|------------|--------------------|------------------------------------------------------------------------|
| profanity  | mute + soften text | Audio muted in subtitle range, subtitle text softened                  |
| drugs      | mute + soften text | Same                                                                   |
| sex        | mute + soften text | Visual matches also trigger `cut`                                      |
| violence   | mute + soften text | Off by default (enable with `--enable-category violence`)              |
| nudity     | cut                | Visual-only, NudeNet                                                   |

Override per-category with `--action <category>=<mute|cut|keep>`.

## Legal

The Family Movie Act of 2005 (US) permits private home performance of filtered authorized copies of motion pictures. cleancut is a tool for personal use on media you own. Don't redistribute the cleaned output.

## License

MIT
