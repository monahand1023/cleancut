from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from cleancut.config import Config
from cleancut.edl import EditDecisionList
from cleancut.pipeline import PipelineOptions, build_edl, render, run_full
from cleancut.probe import (
    audio_streams,
    find_sidecar_subtitle,
    pick_embedded_subtitle,
    probe_streams,
    subtitle_streams,
)
from cleancut.report import build_plan, build_results_report, write_report

console = Console()


def _apply_common(args, config: Config) -> None:
    # Preset goes first so per-flag overrides take precedence.
    if getattr(args, "preset", None):
        config.apply_preset(args.preset)

    if args.wordlists:
        config.override_wordlists(Path(args.wordlists))
    if args.replacements:
        config.override_replacements(Path(args.replacements))
    if args.disable_category:
        for cat in args.disable_category:
            config.enabled_categories.discard(cat)
    if args.enable_category:
        for cat in args.enable_category:
            config.enabled_categories.add(cat)
    for spec in args.action or []:
        if "=" not in spec:
            raise SystemExit(f"--action must be CATEGORY=ACTION, got {spec!r}")
        cat, action = spec.split("=", 1)
        if action not in ("mute", "cut", "keep"):
            raise SystemExit(f"action must be mute|cut|keep, got {action!r}")
        config.actions[cat] = action  # type: ignore[assignment]
    if args.whisper_model:
        config.whisper_model = args.whisper_model
    if args.whisper_device:
        config.whisper_device = args.whisper_device
    if args.whisper_language:
        config.whisper_language = args.whisper_language
    if args.no_word_timestamps:
        config.whisper_word_timestamps = False
    if args.visual_threshold is not None:
        config.visual_threshold = args.visual_threshold
    if args.visual_sample_seconds is not None:
        config.visual_sample_seconds = args.visual_sample_seconds
    if args.visual_min_streak is not None:
        config.visual_min_streak = args.visual_min_streak
    if args.visual_shot_hit_fraction is not None:
        config.visual_shot_hit_fraction = args.visual_shot_hit_fraction
    if args.scene_threshold is not None:
        config.scene_threshold = args.scene_threshold
    if args.no_snap_to_scenes:
        config.snap_cuts_to_scenes = False
    if args.encoder:
        config.encoder = args.encoder
    if args.quality is not None:
        config.quality = args.quality
    if args.density is not None:
        config.density_enabled = args.density
    if args.density_window is not None:
        config.density_window_seconds = args.density_window
    if args.density_min_events is not None:
        config.density_min_events = args.density_min_events
    if args.llm is not None:
        config.llm_enabled = args.llm
    if args.llm_model:
        config.llm_model = args.llm_model
    if args.llm_host:
        config.llm_host = args.llm_host
    if args.llm_min_confidence is not None:
        config.llm_min_confidence = args.llm_min_confidence
    if args.vlm is not None:
        config.vlm_enabled = args.vlm
    if args.vlm_model:
        config.vlm_model = args.vlm_model
    if args.vlm_mode:
        config.vlm_mode = args.vlm_mode
    if args.vlm_stride is not None:
        config.vlm_stride = args.vlm_stride
    if args.vlm_min_confidence is not None:
        config.vlm_min_confidence = args.vlm_min_confidence
    if args.vlm_cut_intimate:
        config.vlm_cut_intimate = True
    if args.vlm_gaps_radius is not None:
        config.vlm_gaps_radius = args.vlm_gaps_radius
    # Track / language selection is on the PipelineOptions, not Config.


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--preset", choices=["fast", "balanced", "thorough"], default="thorough",
        help="Tuning preset. thorough = best quality (default on capable hardware).",
    )
    p.add_argument("--wordlists", help="Path to wordlists JSON (overrides default).")
    p.add_argument("--replacements", help="Path to replacements JSON (overrides default).")
    p.add_argument(
        "--enable-category", action="append",
        choices=["profanity", "drugs", "sex", "violence", "nudity"],
        help="Enable a category (repeatable).",
    )
    p.add_argument(
        "--disable-category", action="append",
        choices=["profanity", "drugs", "sex", "violence", "nudity"],
        help="Disable a category (repeatable).",
    )
    p.add_argument(
        "--action", action="append",
        metavar="CATEGORY=ACTION",
        help="Override action for a category, e.g. profanity=mute (repeatable).",
    )
    p.add_argument(
        "--whisper-model", default=None,
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size.",
    )
    p.add_argument(
        "--whisper-device", default=None, choices=["cpu", "mps", "cuda"],
        help="Force Whisper device. Default: autodetect (MPS on Apple Silicon).",
    )
    p.add_argument("--whisper-language", default=None, help="Force language hint (e.g. 'en').")
    p.add_argument("--no-word-timestamps", action="store_true",
                   help="Disable Whisper word-level timestamps (faster, less precise mutes).")
    p.add_argument("--visual-threshold", type=float, default=None, help="NudeNet confidence threshold (0-1).")
    p.add_argument("--visual-sample-seconds", type=float, default=None, help="Sample 1 frame every N seconds.")
    p.add_argument("--visual-min-streak", type=int, default=None,
                   help="Streak mode: consecutive flagged samples needed to emit a cut.")
    p.add_argument("--visual-shot-hit-fraction", type=float, default=None,
                   help="Shot-aware mode: fraction of sampled frames in a shot that must hit (0-1).")
    p.add_argument("--scene-threshold", type=float, default=None,
                   help="PySceneDetect ContentDetector threshold. Lower = more cuts.")
    p.add_argument("--no-snap-to-scenes", action="store_true",
                   help="Don't snap visual/cut ranges to shot boundaries.")
    p.add_argument("--no-visual", action="store_true", help="Skip the NudeNet visual scan.")
    p.add_argument("--no-whisper", action="store_true", help="Don't transcribe if no .srt is found.")
    p.add_argument("--no-scenes", action="store_true", help="Skip PySceneDetect shot boundary detection.")
    p.add_argument("--no-burn-subs", action="store_true", help="Don't burn the softened subs into the video.")
    p.add_argument("--density", dest="density", action="store_true", default=None,
                   help="Enable density clustering (default: on for thorough preset).")
    p.add_argument("--no-density", dest="density", action="store_false", default=None,
                   help="Disable density clustering.")
    p.add_argument("--density-window", type=float, default=None,
                   help="Rolling window (seconds) for density clustering. Default 60.")
    p.add_argument("--density-min-events", type=int, default=None,
                   help="Minimum hits in window to count as a cluster. Default 3.")
    p.add_argument("--use-llm", dest="llm", action="store_true", default=None,
                   help="Enable local LLM contextual dialogue classification (via Ollama).")
    p.add_argument("--no-llm", dest="llm", action="store_false", default=None,
                   help="Disable LLM dialogue classification.")
    p.add_argument("--llm-model", default=None,
                   help="Ollama model name (default: llama3.1:8b).")
    p.add_argument("--llm-host", default=None,
                   help="Ollama host URL (default: http://127.0.0.1:11434).")
    p.add_argument("--llm-min-confidence", type=float, default=None,
                   help="Discard LLM classifications below this confidence (0-1). Default 0.6.")
    p.add_argument("--use-vlm", dest="vlm", action="store_true", default=None,
                   help="Enable local VLM scene classification (visual gap-closing).")
    p.add_argument("--no-vlm", dest="vlm", action="store_false", default=None,
                   help="Disable VLM scene classification.")
    p.add_argument("--vlm-model", default=None,
                   help="Ollama vision model (default: llava:7b).")
    p.add_argument("--vlm-mode", default=None,
                   choices=["all", "silent", "gaps", "silent+gaps"],
                   help="Which shots to scan. silent+gaps (default) = silent shots + shots near flagged ranges.")
    p.add_argument("--vlm-stride", type=int, default=None,
                   help="For 'all' mode: scan every Nth shot. Default 1.")
    p.add_argument("--vlm-min-confidence", type=float, default=None,
                   help="Discard VLM classifications below this confidence (0-1). Default 0.55.")
    p.add_argument("--vlm-cut-intimate", action="store_true",
                   help="Also cut shots VLM labels 'intimate' (kissing/undressing, no nudity). Off by default.")
    p.add_argument("--vlm-gaps-radius", type=float, default=None,
                   help="In 'gaps' mode: scan shots within N seconds of a flagged range. Default 30.")
    p.add_argument("--encoder", default=None, choices=["auto", "videotoolbox", "libx264"],
                   help="Video encoder. auto = videotoolbox on macOS, libx264 elsewhere.")
    p.add_argument("--quality", type=int, default=None,
                   help="Quality (libx264 CRF; lower=better). Default depends on preset.")
    p.add_argument("--audio-track", type=int, default=None,
                   help="0-indexed audio track to transcribe (audio:0, audio:1, …). Default: prefer English.")
    p.add_argument("--prefer-language", default="eng",
                   help="ISO-639 language code to prefer for subs and audio (default: eng).")
    p.add_argument("--save-transcript", default=None,
                   help="Persist Whisper output to this .srt path (and .words.json alongside).")


def cmd_scan(args) -> int:
    config = Config.load_defaults()
    _apply_common(args, config)
    opts = PipelineOptions(
        video=Path(args.video),
        subs=Path(args.subs) if args.subs else None,
        edl_out=Path(args.output) if args.output else Path(args.video).with_suffix(".edl.json"),
        use_visual=not args.no_visual,
        use_whisper=not args.no_whisper,
        use_scenes=not args.no_scenes,
        audio_track=args.audio_track,
        prefer_language=args.prefer_language,
        save_transcript=Path(args.save_transcript) if args.save_transcript else None,
    )
    edl, _ = build_edl(opts, config)
    edl.to_json(opts.edl_out)
    console.print(f"[green]Wrote EDL[/green] {opts.edl_out}")
    console.print(f"[bold]Summary[/bold]: {edl.summary()}")

    # Always write a human-readable report next to the EDL.
    from cleancut.editor import probe_duration
    try:
        duration = probe_duration(opts.video)
    except Exception:
        duration = None
    report = build_results_report(opts.video, None, edl, original_duration=duration)
    report_path = opts.edl_out.with_suffix(".report.txt")
    write_report(report, report_path)
    console.print(f"[green]Wrote report[/green] {report_path}")
    return 0


def cmd_clean(args) -> int:
    config = Config.load_defaults()
    _apply_common(args, config)
    video = Path(args.video)
    opts = PipelineOptions(
        video=video,
        subs=Path(args.subs) if args.subs else None,
        output=Path(args.output) if args.output else video.with_name(f"{video.stem}.clean{video.suffix}"),
        edl_in=Path(args.edl) if args.edl else None,
        edl_out=Path(args.edl_out) if args.edl_out else None,
        use_visual=not args.no_visual,
        use_whisper=not args.no_whisper,
        use_scenes=not args.no_scenes,
        burn_subs=not args.no_burn_subs,
        audio_track=args.audio_track,
        prefer_language=args.prefer_language,
        save_transcript=Path(args.save_transcript) if args.save_transcript else None,
    )
    out = run_full(opts, config)
    console.print(f"[green bold]Wrote[/green bold] {out}")

    # After clean, write the results report next to the output video.
    # We re-read the EDL that run_full just wrote (if available), otherwise rebuild.
    from cleancut.editor import probe_duration
    edl_for_report: EditDecisionList | None = None
    if opts.edl_out and Path(opts.edl_out).exists():
        edl_for_report = EditDecisionList.from_json(Path(opts.edl_out))
    elif opts.edl_in and Path(opts.edl_in).exists():
        edl_for_report = EditDecisionList.from_json(Path(opts.edl_in))
    if edl_for_report is None:
        # Render didn't persist an EDL — fall back to rebuilding.
        edl_for_report, _ = build_edl(opts, config)
    try:
        duration = probe_duration(opts.video)
    except Exception:
        duration = None
    report = build_results_report(opts.video, opts.output, edl_for_report, original_duration=duration)
    report_path = opts.output.with_suffix(".report.txt")
    write_report(report, report_path)
    console.print(f"[green]Wrote report[/green] {report_path}")
    return 0


def cmd_inspect(args) -> int:
    video = Path(args.video)
    if not video.exists():
        console.print(f"[red]Not found:[/red] {video}")
        return 1
    config = Config.load_defaults()
    _apply_common(args, config)

    streams = probe_streams(video)
    console.print(f"[bold]File:[/bold] {video}")

    audios = audio_streams(streams)
    console.print(f"\n[bold cyan]Audio tracks ({len(audios)})[/bold cyan]")
    for i, s in enumerate(audios):
        ch = f"{s.channels}ch" if s.channels else ""
        console.print(f"  audio:{i}  index={s.index}  lang={s.language}  {s.codec_name}  {ch}  {s.title}")

    subs_streams = subtitle_streams(streams)
    console.print(f"\n[bold cyan]Embedded subtitle tracks ({len(subs_streams)})[/bold cyan]")
    for i, s in enumerate(subs_streams):
        kind = "TEXT" if s.codec_name in {"subrip", "srt", "mov_text", "ass", "ssa", "webvtt"} else "image"
        console.print(f"  sub:{i}    index={s.index}  lang={s.language}  {s.codec_name} [{kind}]  {s.title}")
    embedded = pick_embedded_subtitle(streams, prefer_language=args.prefer_language)
    if embedded:
        console.print(f"  → would extract: stream index {embedded.index} ({embedded.codec_name}, {embedded.language})")

    sidecar = find_sidecar_subtitle(video, prefer_language=args.prefer_language)
    console.print("\n[bold cyan]Sidecar .srt files[/bold cyan]")
    if sidecar:
        console.print(f"  → chosen: {sidecar}")
    for p in sorted(video.parent.glob("*.srt")):
        marker = " ✓" if p == sidecar else ""
        console.print(f"  {p.name}{marker}")

    # The plan — "what would happen if I ran clean now"
    console.print()
    plan = build_plan(
        video=video,
        streams=streams,
        config=config,
        audio_track=args.audio_track,
        prefer_language=args.prefer_language,
        use_visual=not args.no_visual,
        use_scenes=not args.no_scenes,
        use_whisper=not args.no_whisper,
        burn_subs=not args.no_burn_subs,
        explicit_subs=Path(args.subs) if args.subs else None,
    )
    console.print(plan)
    if args.report_out:
        write_report(plan, Path(args.report_out))
        console.print(f"\n[green]Wrote plan[/green] {args.report_out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cleancut", description="Auto-edit movies for content.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", help="Build an EDL without rendering.")
    p_scan.add_argument("video", help="Input video.")
    p_scan.add_argument("--subs", help="External .srt file (else look beside video, else Whisper).")
    p_scan.add_argument("-o", "--output", help="EDL output path (default: <video>.edl.json).")
    _add_common(p_scan)
    p_scan.set_defaults(func=cmd_scan)

    p_clean = sub.add_parser("clean", help="Build an EDL and render the cleaned video.")
    p_clean.add_argument("video", help="Input video.")
    p_clean.add_argument("--subs", help="External .srt file.")
    p_clean.add_argument("-o", "--output", help="Output video (default: <video>.clean.<ext>).")
    p_clean.add_argument("--edl", help="Use an existing EDL JSON instead of scanning.")
    p_clean.add_argument("--edl-out", help="Also write the (re-)built EDL here.")
    _add_common(p_clean)
    p_clean.set_defaults(func=cmd_clean)

    p_inspect = sub.add_parser("inspect", help="Show tracks + a plan of what cleancut would change.")
    p_inspect.add_argument("video", help="Input video.")
    p_inspect.add_argument("--subs", help="External .srt file (overrides discovery).")
    p_inspect.add_argument("--report-out", help="Write the plan to this text file.")
    _add_common(p_inspect)
    p_inspect.set_defaults(func=cmd_inspect)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red bold]error[/red bold]: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
