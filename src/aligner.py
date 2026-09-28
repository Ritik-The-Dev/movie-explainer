"""
aligner.py — decide WHAT the viewer sees during each line of narration.

This is the module that separates a real "movie explained" video from AI slop.
The slop approach writes narration, then slaps evenly-spaced clips under it, so
the picture rarely matches the words. This module does the opposite: every beat
of narration already knows which stretch of the source film it is describing
(the script writer tags it — see the data contract below), and this module's
only job is to turn that stretch into a well-paced sequence of shots that
exactly fills the narration.

    narration says "the camera lingers on a photo on the wall"  (12.3s of audio)
    that beat is tagged  source_span = [6.0, 12.0]  (the real 6s it refers to)
        -> aligner reads the cached shot list for 6.0-12.0
        -> emits: 4.5s live clip of the walk-in  +  a 4s Ken Burns push on the
           photo  +  a 3.8s slow pan across the room  = 12.3s, matched to voice

Two visual primitives, mixed deliberately:

  LIVE CLIP   — a sub-5-second cut of the real footage. Best for motion: a
                punch, a reveal of movement, anything where the frame changing
                IS the point. Kept short both for pace and because short,
                transformed excerpts read very differently from a re-upload.

  FREEZE      — a single frame held with a slow Ken Burns move (zoom/pan). Best
                for dialogue and reveals, where the story is in a face or an
                object, not in movement. Also the workhorse for "copyright-free
                feel": a moving still over narration is transformative in a way
                a raw clip is not.

Why this reads the RAW shot cache, not the beat-sheet windows: the windows in
source_brief_*.json are ~4 minutes wide — far too coarse to cut a 4-second clip
from. The frame-accurate cut points live in source_work/<stem>.shots.json, which
sourceRead already produced. This module consumes those.

Deterministic. Pure stdlib. No ML, no numpy, no network — the actual matching
was already done by the script writer with full context; here we only carve.

Output is an edit-decision list (EDL) as plain JSON, so it can be inspected and
judged on its own before a single frame is rendered.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field

# Windows consoles default to cp1252, which crashes printing non-ASCII
# (Devanagari narration, transcript dialogue). Replace unprintable chars
# with ? instead of dying mid-run.
try:
    if sys.stdout is not None:
        sys.stdout.reconfigure(errors="replace")
except Exception:
    pass


# ---------------------------------------------------------------- tunables

# A live clip never runs longer than this. hp's brief: replay clips stay under
# 5 seconds — for pace, and because short transformed excerpts are a different
# thing from re-uploading a scene.
MAX_CLIP_SECONDS = 5.0

# And never shorter than this, or a clip becomes a flash that reads as an error
# rather than a cut.
MIN_CLIP_SECONDS = 1.6

# A held frame with a Ken Burns move sits in this range. Under ~2s the move has
# no room to breathe; over ~5s a static image starts to feel like the video
# froze. The reveal preset is allowed to run to the top of this band.
MIN_FREEZE_SECONDS = 2.2
MAX_FREEZE_SECONDS = 5.0

# Skip the first fraction of a second of any shot before cutting into it: the
# frame right on a cut can be a dissolve midpoint or a black frame, and starting
# a clip on it looks like a glitch.
SHOT_LEAD_IN = 0.15

# How much of a beat's screen time each style spends on live clips vs freezes.
# The script writer picks the style per beat from its dramatic function; this is
# where that intent becomes a concrete budget.
STYLE_CLIP_FRACTION = {
    "action":   0.75,   # motion is the story — mostly live clips
    "mixed":    0.50,   # default — alternate
    "dialogue": 0.25,   # the story is in faces/words — mostly held frames
    "reveal":   0.15,   # land on the key image and let it breathe
}
DEFAULT_STYLE = "mixed"

# Ken Burns presets, rotated so consecutive freezes don't move identically.
# Each is (zoom_start, zoom_end, pan) — pan is a direction the frame drifts.
# Gentle on purpose: a strong zoom on an explainer looks like a meme, not cinema.
KEN_BURNS_PRESETS = [
    (1.00, 1.12, "in"),      # slow push in
    (1.12, 1.00, "out"),     # slow pull out
    (1.06, 1.10, "lr"),      # drift left-to-right while barely zooming
    (1.06, 1.10, "rl"),      # drift right-to-left
    (1.04, 1.12, "up"),      # rise
    (1.12, 1.06, "down"),    # settle
]


class AlignError(RuntimeError):
    pass


# ---------------------------------------------------------------- data model

@dataclass
class EdlSegment:
    """One visual on screen for `duration` seconds."""
    kind: str                       # "clip" | "freeze"
    src: str                        # source video path
    src_in: float                   # clip: cut-in point; freeze: frame timestamp
    duration: float                 # on-screen seconds
    # Ken Burns params, freeze only.
    zoom_from: float = 1.0
    zoom_to: float = 1.0
    pan: str = "in"
    # Provenance, so a human can audit that the picture matches the words.
    beat: str = ""
    reason: str = ""

    def dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- shot access

def load_shots(source_path: str, work_dir: str) -> list:
    """Load the frame-accurate shot list sourceRead cached for this file.

    The stem logic mirrors sourceRead.MediaInfo.stem so we find the right cache
    even for a --range slice, whose shots file is named with the offset.
    """
    stem = os.path.splitext(os.path.basename(source_path))[0][:60]
    candidates = [
        os.path.join(work_dir, f"{stem}.shots.json"),
    ]
    # A sliced source (season file, --range) has an offset in the stem; match
    # any shots file that starts with the base stem if the exact name is absent.
    for path in candidates:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    # fall back to a prefix match
    if os.path.isdir(work_dir):
        for name in sorted(os.listdir(work_dir)):
            if name.startswith(stem) and name.endswith(".shots.json"):
                with open(os.path.join(work_dir, name), encoding="utf-8") as f:
                    return json.load(f)
    raise AlignError(
        f"no cached shots for {os.path.basename(source_path)} in {work_dir} — "
        f"run sourceRead.py on this file first")


def shots_in_span(shots: list, span_start: float, span_end: float) -> list:
    """Shots overlapping [span_start, span_end], clamped to the span.

    A shot straddling the span boundary is trimmed to the part inside, so a clip
    cut from it can't wander into footage the narration isn't talking about.
    """
    out = []
    for s in shots:
        a = max(s["start"], span_start)
        b = min(s["end"], span_end)
        if b - a <= 0.05:
            continue
        out.append({"start": round(a, 3), "end": round(b, 3),
                    "duration": round(b - a, 3),
                    "orig_duration": s.get("duration", b - a)})
    return out


# ---------------------------------------------------------------- the carve

def _nearest_shot_frame(shots: list, at: float, span_start: float,
                        span_end: float) -> float:
    """A safe frame timestamp near `at` for a freeze.

    Landing a freeze exactly on a cut can catch a dissolve midpoint or black
    frame. If a detected shot contains `at`, return that shot's midpoint (always
    clean); otherwise return `at` clamped into the span.
    """
    for s in shots:
        if s["start"] <= at < s["end"] and s["duration"] > 0.3:
            return round(min(max(s["start"] + s["duration"] * 0.5,
                                 span_start), span_end), 3)
    return round(min(max(at, span_start), span_end), 3)


def align_beat(beat: dict, shots: list, source_path: str) -> list:
    """Turn one narration beat into an EDL that exactly fills its duration.

    beat needs:
      duration     — narration audio length in seconds (from the manifest)
      source_span  — [start, end] absolute source seconds the narration describes
      beat/sceneType (optional) — used only for a human-readable label
      visual_style (optional)   — action|mixed|dialogue|reveal; sets clip/freeze mix

    Model: we have `total` seconds of SCREEN time to fill (the narration length)
    from a SPAN of source footage. A source cursor walks span_start -> span_end
    as segments are emitted, advancing by each segment's screen-duration times
    span_len/total. That guarantees the visuals traverse the whole tagged span
    chronologically no matter the clip/freeze mix, and no matter whether the span
    is longer or shorter than the narration.

      * A CLIP is a continuous real-time slice starting at the cursor — so a run
        of fast cuts in the source simply plays as one energetic clip (which is
        exactly what an action beat wants), rather than being rejected for
        having short shots.
      * A FREEZE samples one clean frame near the cursor and holds it with a Ken
        Burns move — for dialogue/reveal beats where the story is in a face or
        an object, not in movement.

    The clip/freeze *ratio* is set by visual_style; the running ratio is steered
    toward that target, and the FIRST segment already respects the style (a
    reveal opens on a held frame, an action beat opens on a clip). A reveal beat
    is also forced to END on a freeze so the payoff image lands and breathes.
    """
    total = float(beat.get("duration") or 0)
    if total <= 0:
        raise AlignError(f"beat {beat.get('i')} has no narration duration")

    span = beat.get("source_span")
    if not span or len(span) != 2:
        raise AlignError(
            f"beat {beat.get('i')} is not tagged with a source_span — the "
            f"script writer must say which part of the film it describes")
    span_start, span_end = float(span[0]), float(span[1])
    span_len = max(0.1, span_end - span_start)

    label = str(beat.get("beat") or beat.get("sceneType") or f"beat{beat.get('i')}")
    style = str(beat.get("visual_style") or DEFAULT_STYLE).lower()
    clip_fraction = STYLE_CLIP_FRACTION.get(style, STYLE_CLIP_FRACTION[DEFAULT_STYLE])
    reveal_like = style in ("reveal",)

    # Rate at which the source cursor advances per second of screen time, so the
    # segments cover exactly [span_start, span_end] across `total` seconds.
    src_rate = span_len / total

    segments: list[EdlSegment] = []
    remaining = total
    cursor = span_start          # position in the source
    shown_clip = 0.0             # screen-seconds spent on clips so far
    shown_total = 0.0
    preset_idx = 0
    guard = 0

    while remaining > 0.05 and guard < 200:
        guard += 1

        # If too little screen time remains to justify its own segment, fold it
        # into the previous one rather than emitting a sub-second runt (a
        # <2.2s Ken Burns reads as a flash, not a move) — but never fold past
        # the 5s cap: a live excerpt must stay a sub-5-second excerpt, and a
        # held frame past 5s stops breathing and starts dragging. If the cap
        # blocks the fold, fall through and emit the exact tail below.
        if remaining < MIN_FREEZE_SECONDS and segments:
            prev = segments[-1]
            cap = MAX_CLIP_SECONDS if prev.kind == "clip" else MAX_FREEZE_SECONDS
            if prev.duration + remaining <= cap + 1e-9:
                prev.duration = round(prev.duration + remaining, 3)
                if prev.kind == "clip":
                    shown_clip = round(shown_clip + remaining, 3)
                shown_total = round(shown_total + remaining, 3)
                remaining = 0
                break
            # else: fall through — the tail is emitted as its own segment

        # Would this be the final segment? (what's left fits in one piece)
        final = remaining <= MAX_FREEZE_SECONDS + 0.05

        # Pick primitive to steer the running clip-fraction toward the target.
        if shown_total <= 0:
            want_clip = clip_fraction >= 0.5      # first segment respects style
        else:
            want_clip = (shown_clip / shown_total) < clip_fraction
        # A reveal beat must land on a held image.
        if final and reveal_like:
            want_clip = False

        if want_clip:
            # Continuous slice from the cursor. Don't run past the span end.
            room = max(0.0, span_end - cursor)
            dur = min(MAX_CLIP_SECONDS, remaining, max(MIN_CLIP_SECONDS,
                      min(room, MAX_CLIP_SECONDS)))
            dur = min(dur, remaining)
            if room < MIN_CLIP_SECONDS:
                # not enough source left to the right — fall back to a freeze
                want_clip = False
            else:
                seg = EdlSegment(
                    kind="clip", src=source_path, src_in=round(cursor, 3),
                    duration=round(dur, 3), beat=label,
                    reason=f"live slice {cursor:.1f}-{cursor + dur:.1f}s")
                segments.append(seg)
                shown_clip += seg.duration

        if not want_clip:
            if remaining < MIN_FREEZE_SECONDS:
                # Tail the 5s cap refused to absorb: emit it exactly. A short
                # held frame at a beat boundary reads as punctuation, and the
                # beat's total stays narration-locked.
                dur = remaining
            else:
                dur = max(MIN_FREEZE_SECONDS, min(MAX_FREEZE_SECONDS, remaining))
            at = _nearest_shot_frame(shots, cursor + dur * src_rate * 0.5,
                                     span_start, span_end)
            z0, z1, pan = KEN_BURNS_PRESETS[preset_idx % len(KEN_BURNS_PRESETS)]
            preset_idx += 1
            seg = EdlSegment(
                kind="freeze", src=source_path, src_in=at,
                duration=round(dur, 3), zoom_from=z0, zoom_to=z1, pan=pan,
                beat=label, reason=f"held frame @ {at:.1f}s, ken-burns {pan}")
            segments.append(seg)

        shown_total += seg.duration
        remaining = round(remaining - seg.duration, 3)
        # Advance the source cursor so the next segment samples further along.
        cursor = min(span_end, cursor + seg.duration * src_rate)

    # Floating-point cleanup: make the sum exactly equal the narration duration
    # by adjusting the last segment. This is what keeps picture locked to voice.
    drift = round(total - sum(s.duration for s in segments), 3)
    if segments and abs(drift) >= 0.001:
        segments[-1].duration = round(segments[-1].duration + drift, 3)

    return segments


def align_script(beats: list, source_path: str, work_dir: str) -> dict:
    """Align a whole script into one EDL. Returns a dict ready to hand assembly."""
    shots = load_shots(source_path, work_dir)
    edl: list[dict] = []
    for b in beats:
        for seg in align_beat(b, shots, source_path):
            edl.append(seg.dict())

    total = round(sum(s["duration"] for s in edl), 3)
    n_clip = sum(1 for s in edl if s["kind"] == "clip")
    n_freeze = sum(1 for s in edl if s["kind"] == "freeze")
    return {
        "source": source_path,
        "segment_count": len(edl),
        "clips": n_clip,
        "freezes": n_freeze,
        "total_duration": total,
        "segments": edl,
    }


# ---------------------------------------------------------------- CLI

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print("usage:")
        print("  python aligner.py <plan.json> [--work <source_work dir>] "
              "[--out edl.json]")
        print()
        print("plan.json: {source, beats:[{i,beat,duration,source_span,"
              "visual_style}]}")
        sys.exit(0)

    plan_path = sys.argv[1]
    work = "source_work"
    out = None
    if "--work" in sys.argv:
        work = sys.argv[sys.argv.index("--work") + 1]
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]

    with open(plan_path, encoding="utf-8") as f:
        plan = json.load(f)

    result = align_script(plan["beats"], plan["source"], work)
    print(f"  aligned {len(plan['beats'])} beats -> "
          f"{result['segment_count']} segments "
          f"({result['clips']} clips, {result['freezes']} freezes), "
          f"{result['total_duration']:.2f}s")
    for s in result["segments"]:
        tag = "CLIP  " if s["kind"] == "clip" else "FREEZE"
        print(f"    {tag} {s['duration']:>5.2f}s  @src {s['src_in']:>6.2f}s"
              f"  [{s['beat']}]  {s['reason']}")

    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n  edl -> {out}")
