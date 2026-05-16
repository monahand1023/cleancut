"""Density-based clustering of EDL events.

A scene is rarely a single offensive word — it's a *cluster* of them. When N or
more wordlist hits fall within a rolling W-second window, we infer there's a
high-content scene there and emit a single `cut` covering the cluster. Cheap,
no model required, catches "the dealer scene" class of content where context
is dense but each individual hit is borderline.
"""

from __future__ import annotations

from dataclasses import dataclass

from cleancut.edl import EditDecision, EditDecisionList


@dataclass
class DensityParams:
    window_seconds: float = 60.0   # rolling window
    min_events: int = 3            # min hits in window to count as a cluster
    pad_seconds: float = 1.0       # extend cluster by this on each side
    # If the cluster spans more than this, treat it as a "scene" worth cutting.
    min_cluster_span: float = 8.0


def find_clusters(edl: EditDecisionList, params: DensityParams) -> EditDecisionList:
    """Find dense clusters in `edl` and emit one `cut` per cluster.

    Only considers `mute`/`cut` decisions (skips `keep`). Existing cuts inside a
    cluster are absorbed when the new cut is later merged in the pipeline.
    """
    events = sorted(
        [d for d in edl.decisions if d.action in ("mute", "cut") and d.accepted],
        key=lambda d: d.start,
    )
    if len(events) < params.min_events:
        return EditDecisionList(video_path=edl.video_path, subtitle_path=edl.subtitle_path)

    out: list[EditDecision] = []
    n = len(events)
    i = 0
    while i < n:
        # Find the furthest event that starts within window_seconds of events[i].
        window_end = events[i].start + params.window_seconds
        j = i
        while j + 1 < n and events[j + 1].start <= window_end:
            j += 1
        # Number of events in this window.
        in_window = j - i + 1
        if in_window >= params.min_events:
            # Extend the cluster as long as each next event still keeps density.
            cluster_start = events[i].start
            cluster_end = events[j].end
            k = j
            while k + 1 < n:
                # Check if adding events[k+1] keeps a rolling density.
                rolling_start = max(cluster_start, events[k + 1].start - params.window_seconds)
                count = sum(1 for e in events[i:k + 2] if e.start >= rolling_start)
                if count >= params.min_events:
                    k += 1
                    cluster_end = max(cluster_end, events[k].end)
                else:
                    break
            span = cluster_end - cluster_start
            if span >= params.min_cluster_span:
                cats = sorted({e.category.split("+")[0] for e in events[i:k + 1]})
                out.append(
                    EditDecision(
                        start=max(0.0, cluster_start - params.pad_seconds),
                        end=cluster_end + params.pad_seconds,
                        action="cut",
                        category="+".join(cats),
                        reason=f"density: {k - i + 1} events in {span:.1f}s window",
                        source="density",
                    )
                )
            i = k + 1
        else:
            i += 1

    return EditDecisionList(decisions=out, video_path=edl.video_path, subtitle_path=edl.subtitle_path)
