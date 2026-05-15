from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from cleancut.config import Config
from cleancut.edl import EditDecisionList
from cleancut.pipeline import PipelineOptions, build_edl, render, run_full

console = Console()


def _apply_common(args, config: Config) -> None:
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
    if args.visual_threshold is not None:
        config.visual_threshold = args.visual_threshold
    if args.visual_sample_seconds is not None:
        config.visual_sample_seconds = args.visual_sample_seconds


def _add_common(p: argparse.ArgumentParser) -> None:
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
    p.add_argument("--whisper-model", default=None, choices=["tiny", "base", "small", "medium", "large"])
    p.add_argument("--visual-threshold", type=float, default=None, help="NudeNet confidence threshold (0-1).")
    p.add_argument("--visual-sample-seconds", type=float, default=None, help="Sample 1 frame every N seconds.")
    p.add_argument("--no-visual", action="store_true", help="Skip the NudeNet visual scan.")
    p.add_argument("--no-whisper", action="store_true", help="Don't transcribe if no .srt is found.")
    p.add_argument("--no-burn-subs", action="store_true", help="Don't burn the softened subs into the video.")


def cmd_scan(args) -> int:
    config = Config.load_defaults()
    _apply_common(args, config)
    opts = PipelineOptions(
        video=Path(args.video),
        subs=Path(args.subs) if args.subs else None,
        edl_out=Path(args.output) if args.output else Path(args.video).with_suffix(".edl.json"),
        use_visual=not args.no_visual,
        use_whisper=not args.no_whisper,
    )
    edl, _ = build_edl(opts, config)
    edl.to_json(opts.edl_out)
    console.print(f"[green]Wrote EDL[/green] {opts.edl_out}")
    console.print(f"[bold]Summary[/bold]: {edl.summary()}")
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
        burn_subs=not args.no_burn_subs,
    )
    out = run_full(opts, config)
    console.print(f"[green bold]Wrote[/green bold] {out}")
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
