"""Full pipeline: video + subs -> EDL -> cleaned video."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from cleancut.config import Config
from cleancut.edl import EditDecision, EditDecisionList
from cleancut.editor import (
    Range,
    adjust_subtitles_for_cuts,
    apply_cuts,
    apply_mutes_and_subs,
    edl_to_ranges,
    shift_ranges_after_cuts,
)
from cleancut.scenes import Shot, snap_range_to_shots
from cleancut.subtitles import (
    Subtitle,
    read_srt,
    scan_subtitles,
    scan_words,
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
    use_scenes: bool = True
    burn_subs: bool = True
    work_dir: Path | None = None
    # 0-indexed audio track ordinal (audio:0, audio:1, …); None = auto (prefer English).
    audio_track: int | None = None
    # ISO language code to prefer for both sidecar .srt and audio track ("eng", "spa", …).
    prefer_language: str = "eng"
    # If set, persist Whisper output to this .srt path (and .words.json alongside).
    save_transcript: Path | None = None


def _get_subtitles_and_words(
    opts: PipelineOptions, config: Config
) -> tuple[list[Subtitle], list]:
    """Resolve subtitles in this priority:

    1. Explicit --subs path.
    2. Best sidecar .srt found beside the video (Plex naming, language-aware).
    3. Best embedded *text* subtitle track in the container.
    4. Whisper transcription of the preferred-language audio track.
    """
    from cleancut.probe import (
        extract_audio_to_wav,
        extract_text_subtitle,
        find_sidecar_subtitle,
        pick_audio_track,
        pick_embedded_subtitle,
        probe_streams,
    )

    if opts.subs and opts.subs.exists():
        console.print(f"[cyan]Reading subtitles[/cyan] {opts.subs}")
        return read_srt(opts.subs), []

    sidecar = find_sidecar_subtitle(opts.video, prefer_language=opts.prefer_language)
    if sidecar:
        console.print(f"[cyan]Found sidecar .srt[/cyan] {sidecar}")
        return read_srt(sidecar), []

    streams = probe_streams(opts.video)
    embedded = pick_embedded_subtitle(streams, prefer_language=opts.prefer_language)
    if embedded:
        console.print(
            f"[cyan]Extracting embedded {embedded.codec_name} subtitle "
            f"(lang={embedded.language})[/cyan]"
        )
        extracted = extract_text_subtitle(opts.video, embedded.index)
        if extracted:
            return read_srt(extracted), []

    if not opts.use_whisper:
        console.print("[yellow]No subtitles and Whisper disabled — skipping dialogue scan.[/yellow]")
        return [], []

    # Pick which audio track to feed Whisper.
    try:
        track = pick_audio_track(streams, opts.audio_track, prefer_language=opts.prefer_language)
    except ValueError as e:
        raise SystemExit(str(e))
    if track is None:
        console.print("[yellow]No audio tracks found — skipping dialogue scan.[/yellow]")
        return [], []

    console.print(
        f"[cyan]Using audio track[/cyan] index={track.index} lang={track.language} "
        f"codec={track.codec_name}"
    )
    audio_path = extract_audio_to_wav(opts.video, track.index)

    from cleancut.transcribe import _resolve_device, transcribe

    device = _resolve_device(config.whisper_device, config.whisper_word_timestamps)
    console.print(
        f"[cyan]Transcribing with Whisper[/cyan] model={config.whisper_model} "
        f"device={device} word_timestamps={config.whisper_word_timestamps}"
    )
    try:
        subs, words = transcribe(
            opts.video,
            model_name=config.whisper_model,
            device=device,
            word_timestamps=config.whisper_word_timestamps,
            language=config.whisper_language,
            audio_path=audio_path,
        )
    finally:
        audio_path.unlink(missing_ok=True)

    # Persist the transcript so subsequent scans can reuse it without re-running Whisper.
    if opts.save_transcript:
        srt_out = Path(opts.save_transcript)
        srt_out.parent.mkdir(parents=True, exist_ok=True)
        write_srt(subs, srt_out)
        console.print(f"[green]Saved transcript[/green] {srt_out}")
        if words:
            words_out = srt_out.with_suffix(".words.json")
            import json
            words_out.write_text(json.dumps(
                [{"start": w.start, "end": w.end, "text": w.text, "probability": w.probability}
                 for w in words], indent=2,
            ))
            console.print(f"[green]Saved word timings[/green] {words_out}")

    return subs, words


def _detect_scenes_if_enabled(opts: PipelineOptions, config: Config) -> list[Shot]:
    if not opts.use_scenes:
        return []
    try:
        from cleancut.scenes import detect_shots
    except RuntimeError as e:
        console.print(f"[yellow]Scene detection skipped: {e}[/yellow]")
        return []
    console.print(f"[cyan]Detecting shot boundaries[/cyan] (threshold={config.scene_threshold})")
    try:
        shots = detect_shots(opts.video, threshold=config.scene_threshold)
        console.print(f"[green]Found {len(shots)} shots[/green]")
        return shots
    except Exception as e:
        console.print(f"[yellow]Scene detection failed: {e}[/yellow]")
        return []


def _snap_edl_to_shots(edl: EditDecisionList, shots: list[Shot]) -> EditDecisionList:
    """Extend each `cut` decision outward to enclosing shot boundaries.

    Mutes are left alone — they're audio-only and should be word-precise.
    """
    if not shots:
        return edl
    out: list[EditDecision] = []
    for d in edl.decisions:
        if d.action == "cut":
            ns, ne = snap_range_to_shots(d.start, d.end, shots)
            out.append(
                EditDecision(
                    start=ns,
                    end=ne,
                    action=d.action,
                    category=d.category,
                    reason=(d.reason + " | snapped-to-shot").strip(" |"),
                    text_before=d.text_before,
                    text_after=d.text_after,
                    source=d.source,
                    accepted=d.accepted,
                )
            )
        else:
            out.append(d)
    return EditDecisionList(decisions=out, video_path=edl.video_path, subtitle_path=edl.subtitle_path)


def build_edl(opts: PipelineOptions, config: Config) -> tuple[EditDecisionList, list[Subtitle]]:
    """Run all detectors and produce a merged EDL."""
    subs, words = _get_subtitles_and_words(opts, config)

    edl = EditDecisionList(video_path=str(opts.video), subtitle_path=str(opts.subs or ""))

    # Dialogue scan: prefer word-level if Whisper gave us words, else line-level.
    if words:
        console.print(f"[cyan]Scanning {len(words)} words (word-precise)[/cyan]")
        edl.extend(scan_words(words, config).decisions)
    elif subs:
        console.print(f"[cyan]Scanning {len(subs)} subtitle lines[/cyan]")
        edl.extend(scan_subtitles(subs, config).decisions)

    # Shot boundaries (also used for shot-aware visual scan below).
    shots = _detect_scenes_if_enabled(opts, config)

    if opts.use_visual and "nudity" in config.enabled_categories:
        try:
            from cleancut.visual import scan_video
            console.print(
                f"[cyan]Visual scan[/cyan] "
                f"({'shot-aware' if shots else 'streak mode'}, "
                f"sample={config.visual_sample_seconds}s, "
                f"threshold={config.visual_threshold})"
            )
            edl.extend(scan_video(opts.video, config, shots=shots or None).decisions)
        except RuntimeError as e:
            console.print(f"[yellow]Visual scan skipped: {e}[/yellow]")

    edl = edl.pad(config.pad_seconds).merge_overlapping(gap=config.merge_gap_seconds).sorted()

    # Density clustering — promote dense clusters of small hits into a single cut.
    if config.density_enabled and len(edl) >= config.density_min_events:
        from cleancut.density import DensityParams, find_clusters
        console.print("[cyan]Density clustering[/cyan] over dialogue/visual hits")
        clusters = find_clusters(
            edl,
            DensityParams(
                window_seconds=config.density_window_seconds,
                min_events=config.density_min_events,
                min_cluster_span=config.density_min_cluster_span,
            ),
        )
        if len(clusters):
            console.print(f"[green]Density found {len(clusters)} cluster(s)[/green]")
            edl.extend(clusters.decisions)

    # LLM-based contextual dialogue classification.
    if config.llm_enabled and subs:
        try:
            from cleancut.classify_dialogue import LLMParams, classify_dialogue
            console.print(f"[cyan]LLM dialogue scan[/cyan] model={config.llm_model}")
            llm_edl = classify_dialogue(
                subs,
                LLMParams(
                    model=config.llm_model,
                    ollama_host=config.llm_host,
                    min_confidence=config.llm_min_confidence,
                ),
            )
            if len(llm_edl):
                console.print(f"[green]LLM flagged {len(llm_edl)} scene(s)[/green]")
                edl.extend(llm_edl.decisions)
        except RuntimeError as e:
            console.print(f"[yellow]LLM scan skipped: {e}[/yellow]")

    # Audio event detection (AST) — catches moans/screams/gunshots without dialogue or visible content.
    if config.audio_events_enabled and shots:
        try:
            from cleancut.audio_events import AudioEventParams, scan_audio_events
            console.print(f"[cyan]Audio event scan[/cyan] model={config.audio_events_model}")
            ae_edl = scan_audio_events(
                opts.video, shots,
                AudioEventParams(
                    model=config.audio_events_model,
                    threshold=config.audio_events_threshold,
                    clip_seconds=config.audio_events_clip_seconds,
                    skip_violence=config.audio_events_skip_violence,
                ),
                audio_track_index=None,  # use first audio track by default; CLI can override
            )
            if len(ae_edl):
                console.print(f"[green]Audio events flagged {len(ae_edl)} shot(s)[/green]")
                edl.extend(ae_edl.decisions)
        except RuntimeError as e:
            console.print(f"[yellow]Audio events skipped: {e}[/yellow]")

    # VLM visual scene classification — closes the gap on silent scenes.
    if config.vlm_enabled and shots:
        try:
            from cleancut.classify_visual import VLMParams, scan_video as vlm_scan
            cut_on = ("intimate", "explicit", "drug_use", "violence") if config.vlm_cut_intimate \
                else ("explicit", "drug_use", "violence")
            console.print(
                f"[cyan]VLM scene scan[/cyan] model={config.vlm_model} "
                f"mode={config.vlm_mode}"
            )
            vlm_edl = vlm_scan(
                opts.video, shots, subs, edl,
                VLMParams(
                    model=config.vlm_model,
                    mode=config.vlm_mode,
                    stride=config.vlm_stride,
                    min_confidence=config.vlm_min_confidence,
                    gaps_radius_seconds=config.vlm_gaps_radius,
                    cut_on=cut_on,
                    ollama_host=config.llm_host,
                ),
            )
            if len(vlm_edl):
                console.print(f"[green]VLM flagged {len(vlm_edl)} shot(s)[/green]")
                edl.extend(vlm_edl.decisions)
        except RuntimeError as e:
            console.print(f"[yellow]VLM scan skipped: {e}[/yellow]")

    # Re-merge after adding density/LLM/VLM signals.
    edl = edl.merge_overlapping(gap=config.merge_gap_seconds).sorted()
    if shots and config.snap_cuts_to_scenes:
        edl = _snap_edl_to_shots(edl, shots)
        edl = edl.merge_overlapping(gap=0.0).sorted()

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

    # Default work dir alongside the OUTPUT, not the source — source dir may have
    # weird characters that break ffmpeg filters, or may be on a read-only volume.
    work = opts.work_dir or (opts.output.parent if opts.output else opts.video.parent) / ".cleancut_work"
    work.mkdir(parents=True, exist_ok=True)

    cuts = edl_to_ranges(edl, "cut")
    mutes = edl_to_ranges(edl, "mute")
    encoder = config.resolved_encoder()
    console.print(f"[cyan]Encoder[/cyan]: {encoder} (q={config.quality})")

    # Step 1: apply cuts (re-encode if needed).
    if cuts:
        cut_path = work / f"{opts.video.stem}.cut.mp4"
        console.print(f"[cyan]Applying {len(cuts)} cut(s)…[/cyan]")
        apply_cuts(opts.video, cuts, cut_path, encoder=encoder, quality=config.quality)
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
        encoder=encoder,
        quality=config.quality,
    )

    return opts.output


def run_full(opts: PipelineOptions, config: Config) -> Path:
    if opts.edl_in:
        console.print(f"[cyan]Loading EDL from[/cyan] {opts.edl_in}")
        edl = EditDecisionList.from_json(opts.edl_in)
        subs, _ = _get_subtitles_and_words(opts, config)
    else:
        edl, subs = build_edl(opts, config)

    if opts.edl_out:
        edl.to_json(opts.edl_out)
        console.print(f"[green]Wrote EDL[/green] {opts.edl_out}")

    console.print(f"[bold]EDL summary[/bold]: {edl.summary()}")

    if opts.output:
        return render(edl, subs, opts, config)
    return opts.edl_out or Path()
