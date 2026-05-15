"""Full pipeline: video + subs -> EDL -> cleaned video."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rich.console import Console

from cleancut.config import Config
from cleancut.edl import EditDecisionList
from cleancut.editor import (
    Range,
    adjust_subtitles_for_cuts,
    apply_cuts,
    apply_mutes_and_subs,
    edl_to_ranges,
    shift_ranges_after_cuts,
)
from cleancut.subtitles import (
    Subtitle,
    read_srt,
    scan_subtitles,
    softened_subtitles,
    write_srt,
)

console = Console()


@dataclass
class PipelineOptions:
    video: Path
    subs: Path | None = None
    output: Path | None = None
    edl_in: Path | None = None
    edl_out: Path | None = None
    use_visual: bool = True
    use_whisper: bool = True
    burn_subs: bool = True
    work_dir: Path | None = None


def get_subtitles(opts: PipelineOptions, config: Config) -> list[Subtitle]:
    """Read .srt if provided, else look next to video, else transcribe with Whisper."""
    if opts.subs and opts.subs.exists():
        console.print(f"[cyan]Reading subtitles[/cyan] {opts.subs}")
        return read_srt(opts.subs)

    # Look for sibling .srt
    sibling = opts.video.with_suffix(".srt")
    if sibling.exists():
        console.print(f"[cyan]Found sibling .srt[/cyan] {sibling}")
        return read_srt(sibling)

    if not opts.use_whisper:
        console.print("[yellow]No subtitles and Whisper disabled — skipping dialogue scan.[/yellow]")
        return []

    console.print(f"[cyan]Transcribing with Whisper ({config.whisper_model})[/cyan] — this may take a while…")
    from cleancut.transcribe import transcribe
    return transcribe(opts.video, config.whisper_model)


def build_edl(opts: PipelineOptions, config: Config) -> tuple[EditDecisionList, list[Subtitle]]:
    """Run all detectors and produce a merged EDL."""
    subs = get_subtitles(opts, config)

    edl = EditDecisionList(video_path=str(opts.video), subtitle_path=str(opts.subs or ""))

    if subs:
        console.print(f"[cyan]Scanning {len(subs)} subtitle lines[/cyan]")
        edl.extend(scan_subtitles(subs, config).decisions)

    if opts.use_visual and "nudity" in config.enabled_categories:
        console.print("[cyan]Visual scan (NudeNet)[/cyan] — this is slow…")
        try:
            from cleancut.visual import scan_video
            edl.extend(scan_video(opts.video, config).decisions)
        except RuntimeError as e:
            console.print(f"[yellow]Visual scan skipped: {e}[/yellow]")

    edl = edl.pad(config.pad_seconds).merge_overlapping(gap=config.merge_gap_seconds).sorted()
    return edl, subs


def render(
    edl: EditDecisionList,
    subs: list[Subtitle],
    opts: PipelineOptions,
    config: Config,
) -> Path:
    """Apply the EDL to the video. Returns the path to the cleaned video."""
    if not opts.output:
        raise ValueError("output path required for render")

    work = opts.work_dir or opts.video.parent / ".cleancut_work"
    work.mkdir(parents=True, exist_ok=True)

    cuts = edl_to_ranges(edl, "cut")
    mutes = edl_to_ranges(edl, "mute")

    # Step 1: apply cuts (re-encode if needed).
    if cuts:
        cut_path = work / f"{opts.video.stem}.cut.mp4"
        console.print(f"[cyan]Applying {len(cuts)} cut(s)…[/cyan]")
        apply_cuts(opts.video, cuts, cut_path)
        # Mutes and subtitles need to shift onto the cut timeline.
        mutes = shift_ranges_after_cuts(mutes, cuts)
        subs = adjust_subtitles_for_cuts(subs, cuts)
    else:
        cut_path = opts.video

    # Step 2: soften surviving subs and write a working .srt for burn-in.
    softened = softened_subtitles(subs, config) if subs else []
    srt_for_burn: Path | None = None
    if softened and opts.burn_subs:
        srt_for_burn = work / f"{opts.video.stem}.softened.srt"
        write_srt(softened, srt_for_burn)

    # Step 3: mute audio ranges and burn the softened subs.
    console.print(
        f"[cyan]Muting {len(mutes)} range(s){' + burning subs' if srt_for_burn else ''}[/cyan]"
    )
    apply_mutes_and_subs(
        input_path=cut_path,
        mutes=mutes,
        srt_path=srt_for_burn,
        output_path=opts.output,
        burn_subs=bool(srt_for_burn),
    )

    return opts.output


def run_full(opts: PipelineOptions, config: Config) -> Path:
    if opts.edl_in:
        console.print(f"[cyan]Loading EDL from[/cyan] {opts.edl_in}")
        edl = EditDecisionList.from_json(opts.edl_in)
        subs = get_subtitles(opts, config)
    else:
        edl, subs = build_edl(opts, config)

    if opts.edl_out:
        edl.to_json(opts.edl_out)
        console.print(f"[green]Wrote EDL[/green] {opts.edl_out}")

    console.print(f"[bold]EDL summary[/bold]: {edl.summary()}")

    if opts.output:
        return render(edl, subs, opts, config)
    return opts.edl_out or Path()
