"""
assemble.py — render an aligner EDL into a finished explainer video.

Takes three things and produces one MP4:
  * the EDL from aligner.py      — what to show, and for exactly how long
  * the voiceover manifest        — the narration audio, beat by beat
  * the source film               — where live clips and freeze frames come from

Design rules, each earned:

  1. PICTURE IS BUILT PER BEAT, IN LOCKSTEP WITH ITS OWN AUDIO — never against
     absolute manifest timestamps. The manifest's `timeline` accumulates only
     beat durations; it does NOT include the ~0.95s paragraph pauses that
     voiceover.stitch() bakes between beats in narration_full.wav. So the Nth
     beat's real position in the full mix is later than timeline[N].start by the
     sum of all prior pauses. Laying video at those timestamps would drift ~1s
     per beat. Instead we render each beat's visuals to exactly match that
     beat's OWN audio file, then concatenate video and audio the same way, with
     the same pauses. Sync cannot drift because the two tracks are cut from the
     same ruler.

  2. PURE FFMPEG SUBPROCESS. No moviepy, no opencv, no numpy. On this machine
     Python CV/DSP libraries have repeatedly dragged numpy off its 1.22 pin and
     broken the TTS stack; ffmpeg is already installed and does all of this
     natively. (This mirrors the same decision sourceRead.py made for shots.)

  3. GPU ENCODE WHEN AVAILABLE. The RX 6500M can encode H.264/HEVC via AMD's
     AMF. We probe for h264_amf and use it; otherwise libx264. The ML stages are
     CPU-bound here (AMD, no CUDA), so the encoder is the one place the GPU
     helps, and it helps most on the final full-length render.

Everything is normalised to one canonical size / fps / pixel format before
concat, because concat demuxer refuses to join streams that don't match, and a
freeze rendered from a still won't share timebase with a clip cut from 24fps
footage unless we force it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

# Windows consoles default to cp1252, which crashes printing non-ASCII
# (Devanagari narration, transcript dialogue). Replace unprintable chars
# with ? instead of dying mid-run.
try:
    if sys.stdout is not None:
        sys.stdout.reconfigure(errors="replace")
except Exception:
    pass


# ---------------------------------------------------------------- canonical format

# Everything is conformed to this before concat. 1080p30 is the sweet spot for
# a talking-over-footage explainer: sharp enough to look deliberate, cheap
# enough to render a full film overnight on this hardware.
OUT_W = int(os.getenv("ASSEMBLE_W", "1920"))
OUT_H = int(os.getenv("ASSEMBLE_H", "1080"))
FPS = int(os.getenv("ASSEMBLE_FPS", "30"))
PIX = "yuv420p"
AUDIO_SR = 24000        # matches voiceover output; no resample needed

# Pause baked between beats. MUST equal voiceover.PAUSE_AFTER["paragraph"], or
# video and audio beat boundaries diverge. Imported if possible, else this
# literal — kept in sync deliberately.
try:
    from audioPolish import PAUSE_AFTER
    BEAT_PAUSE = float(PAUSE_AFTER["paragraph"])
except Exception:
    BEAT_PAUSE = 0.95


class AssembleError(RuntimeError):
    pass


# ---------------------------------------------------------------- ffmpeg helpers

def _run(cmd: list, *, desc: str = "") -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-8:]
        raise AssembleError(
            f"ffmpeg failed{' during ' + desc if desc else ''} "
            f"(exit {proc.returncode}):\n  " + "\n  ".join(tail))


def probe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True)
    try:
        return round(float(out.stdout.strip()), 3)
    except ValueError:
        return 0.0


def pick_encoder() -> list:
    """Prefer the AMD hardware encoder, fall back to libx264 — but only after
    proving the chosen encoder can actually INITIALIZE on this machine.

    Being listed by `ffmpeg -encoders` means the encoder was compiled in, NOT
    that the hardware/driver is present. In this sandbox h264_qsv is listed but
    has no Intel silicon behind it and dies with "unsupported" the moment it's
    used. On hp's box h264_amf could likewise be listed but fail if the AMD
    driver isn't cooperating. Discovering that mid-render at 6am wastes the whole
    night, so we do a 1-frame smoke encode of each candidate and take the first
    that truly works. The video filters run on CPU either way; only the final
    encode is offloaded.
    """
    forced = os.getenv("ASSEMBLE_ENCODER", "").strip()
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                              capture_output=True, text=True).stdout

    def listed(name: str) -> bool:
        return name in encoders

    def works(name: str) -> bool:
        """1-frame smoke test: can this encoder actually open?"""
        try:
            proc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "lavfi", "-i", f"color=c=black:s=320x240:r={FPS}",
                 "-frames:v", "1", *_encoder_args(name),
                 "-f", "null", "-"],
                capture_output=True, text=True, timeout=30)
            return proc.returncode == 0
        except Exception:
            return False

    candidates = []
    if forced:
        candidates.append(forced)
    candidates += ["h264_amf", "h264_nvenc", "h264_qsv", "libx264"]

    for name in candidates:
        if not listed(name):
            continue
        if name == "libx264" or works(name):
            hw = "software" if name == "libx264" else "hardware, probe-verified"
            print(f"  encoder: {name} ({hw})")
            return _encoder_args(name)
        else:
            print(f"  encoder: {name} listed but failed to initialize — skipping")

    print(f"  encoder: libx264 (software fallback)")
    return _encoder_args("libx264")


def _encoder_args(name: str) -> list:
    if name == "h264_amf":
        # AMF quality knobs differ from x264's. quality=quality is the slow,
        # good preset; rc=cqp with qp ~22 is visually clean for this content.
        return ["-c:v", "h264_amf", "-quality", "quality", "-rc", "cqp",
                "-qp_i", "22", "-qp_p", "22", "-pix_fmt", PIX]
    if name == "h264_qsv":
        return ["-c:v", "h264_qsv", "-global_quality", "22", "-pix_fmt", PIX]
    if name == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-rc", "vbr", "-cq", "22",
                "-preset", "p5", "-pix_fmt", PIX]
    # libx264: crf 20 is visually lossless-ish for this kind of content;
    # veslow would be overkill overnight, medium is the sensible default.
    return ["-c:v", "libx264", "-crf", "20", "-preset", "medium",
            "-pix_fmt", PIX]


# ---------------------------------------------------------------- primitives

def _pan_expr(pan: str, frames: int):
    """(x_expr, y_expr) for zoompan given a pan direction.

    zoompan evaluates x/y each output frame; iw/ih are the (already upscaled)
    input dims, iw/zoom the visible width. Centre is the baseline; the presets
    nudge the window across the frame over the clip's life using `on`
    (output frame index) against the total frame count.

    NOTE: `d` is a zoompan *option*, not an expression variable — using
    `on/d` in an expression fails config with "Undefined constant". The
    numeric frame count must be baked in.
    """
    cx = "(iw-iw/zoom)/2"
    cy = "(ih-ih/zoom)/2"
    if pan == "lr":
        return (f"(iw-iw/zoom)*(on/{frames})", cy)
    if pan == "rl":
        return (f"(iw-iw/zoom)*(1-on/{frames})", cy)
    if pan == "up":
        return (cx, f"(ih-ih/zoom)*(1-on/{frames})")
    if pan == "down":
        return (cx, f"(ih-ih/zoom)*(on/{frames})")
    return (cx, cy)     # "in"/"out" — pure zoom, stay centred


def render_freeze(seg: dict, dst: str) -> None:
    """A single source frame held for seg.duration with a Ken Burns move.

    We grab the frame, then drive zoompan for exactly duration*FPS frames. The
    still is pre-upscaled 2x before zoompan so the zoom never reveals softness
    (zoompan interpolates, and interpolating an already-full-res frame shimmers).
    """
    frames = max(1, int(round(seg["duration"] * FPS)))
    z0 = float(seg.get("zoom_from", 1.0))
    z1 = float(seg.get("zoom_to", 1.0))
    # zoompan wants a single z expression evaluated per frame. Linear from z0->z1
    # across `frames`. Clamp so rounding never sends zoom below 1.0 (which would
    # show background).
    z_expr = f"max(1.001,{z0}+({z1}-{z0})*on/{max(1, frames-1)})"
    xex, yex = _pan_expr(seg.get("pan", "in"), frames)

    # Upscale the source frame first (scale to 2x canonical), then zoompan down
    # to canonical. s= is the OUTPUT size of zoompan.
    up_w, up_h = OUT_W * 2, OUT_H * 2
    vf = (
        f"scale={up_w}:{up_h}:force_original_aspect_ratio=increase,"
        f"crop={up_w}:{up_h},"
        f"zoompan=z='{z_expr}':x='{xex}':y='{yex}':d={frames}:"
        f"s={OUT_W}x{OUT_H}:fps={FPS},"
        f"setsar=1"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{seg['src_in']:.3f}", "-i", seg["src"],
        "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-",  # placeholder
    ]
    # The one-frame grab and the zoompan are cleaner as two steps: grab a PNG,
    # then animate it. Piping is fragile across ffmpeg versions.
    tmp_png = dst + ".frame.png"
    _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
          "-ss", f"{seg['src_in']:.3f}", "-i", seg["src"],
          "-frames:v", "1", tmp_png], desc="freeze frame grab")
    _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
          "-loop", "1", "-i", tmp_png, "-t", f"{seg['duration']:.3f}",
          "-vf", vf, "-r", str(FPS), "-an",
          "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
          "-pix_fmt", PIX, dst], desc="freeze render")
    if os.path.exists(tmp_png):
        os.remove(tmp_png)


def render_clip(seg: dict, dst: str) -> None:
    """A live cut of the source, conformed to canonical size/fps/pixfmt.

    Letterbox rather than crop: an explainer that crops faces to fill 16:9 looks
    careless. force_original_aspect_ratio=decrease + pad keeps the whole frame.
    """
    vf = (
        f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
        f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setsar=1,fps={FPS}"
    )
    _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{seg['src_in']:.3f}", "-i", seg["src"],
        "-t", f"{seg['duration']:.3f}",
        "-vf", vf, "-an",
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
        "-pix_fmt", PIX, dst,
    ], desc="clip render")


# ---------------------------------------------------------------- beat grouping

def _group_by_beat(edl_segments: list, timeline: list) -> list:
    """Attach each EDL segment to its beat, in manifest order.

    The EDL carries a `beat` label on every segment; the manifest timeline lists
    beats in order with their audio files. We group EDL segments by their beat
    label and pair them with the matching audio, preserving order.
    """
    # Map beat label -> audio file + duration, in order.
    order = []
    by_label: dict = {}
    for t in timeline:
        label = str(t.get("beat"))
        order.append(label)
        by_label[label] = t

    groups = []
    # Walk the EDL once; whenever the beat label changes, start a new group.
    cur_label = None
    cur_segs: list = []
    for seg in edl_segments:
        lbl = str(seg.get("beat"))
        if lbl != cur_label and cur_segs:
            groups.append((cur_label, cur_segs))
            cur_segs = []
        cur_label = lbl
        cur_segs.append(seg)
    if cur_segs:
        groups.append((cur_label, cur_segs))

    # Pair each group with its audio by label; fall back to positional if labels
    # don't line up (a hand-written test plan may not match manifest labels).
    paired = []
    for i, (label, segs) in enumerate(groups):
        audio = by_label.get(label)
        if audio is None and i < len(timeline):
            audio = timeline[i]
        if audio is None:
            raise AssembleError(
                f"beat group {i} ({label!r}) has no matching audio in the "
                f"manifest timeline")
        paired.append({"label": label, "segments": segs, "audio": audio})
    return paired


# ---------------------------------------------------------------- assembly

def _concat(paths: list, dst: str, work: str, *, kind: str) -> None:
    """Concat a list of same-format files via the concat demuxer."""
    listing = os.path.join(work, f"concat_{kind}.txt")
    with open(listing, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    args = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", listing]
    if kind == "video":
        args += ["-c", "copy", dst]
    else:
        args += ["-c", "copy", dst]
    _run(args, desc=f"{kind} concat")


def _silence(duration: float, dst: str) -> None:
    _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
          "-f", "lavfi", "-i", f"anullsrc=r={AUDIO_SR}:cl=mono",
          "-t", f"{duration:.3f}", "-c:a", "pcm_s16le", dst],
         desc="silence")


def _black(duration: float, dst: str) -> None:
    """A black video segment (fills the audio pause between beats)."""
    _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
          "-f", "lavfi", "-i",
          f"color=c=black:s={OUT_W}x{OUT_H}:r={FPS}",
          "-t", f"{duration:.3f}", "-pix_fmt", PIX,
          "-c:v", "libx264", "-crf", "24", "-preset", "veryfast", dst],
         desc="black filler")


def assemble(edl: dict, manifest: dict, out_path: str, *,
             work_dir: str | None = None, keep_work: bool = False) -> dict:
    """Render EDL + narration into a finished video. Returns a small report."""
    timeline = manifest.get("timeline") or []
    if not timeline:
        raise AssembleError("manifest has no timeline")
    segments = edl.get("segments") or []
    if not segments:
        raise AssembleError("EDL has no segments")

    work = work_dir or tempfile.mkdtemp(prefix="assemble_")
    os.makedirs(work, exist_ok=True)
    made_temp = work_dir is None

    try:
        groups = _group_by_beat(segments, timeline)
        print(f"  {len(groups)} beats, {len(segments)} visual segments")

        video_parts: list = []
        audio_parts: list = []

        for gi, group in enumerate(groups):
            segs = group["segments"]
            audio = group["audio"]
            audio_file = audio.get("file")
            if not audio_file or not os.path.exists(audio_file):
                raise AssembleError(
                    f"beat {gi} audio missing: {audio_file!r}")

            beat_audio_dur = probe_duration(audio_file)

            # Render this beat's visuals.
            rendered = []
            for si, seg in enumerate(segs):
                part = os.path.join(work, f"b{gi:02d}_s{si:02d}.mp4")
                if seg["kind"] == "freeze":
                    render_freeze(seg, part)
                else:
                    render_clip(seg, part)
                rendered.append(part)

            # Concat this beat's visuals into one video, then correct any tiny
            # drift between the visual total and the actual audio length by
            # trimming/padding the beat's video to the audio. This is the sync
            # guarantee: each beat's picture is made exactly as long as its own
            # narration, so nothing accumulates.
            beat_vid_raw = os.path.join(work, f"beat{gi:02d}_raw.mp4")
            _concat(rendered, beat_vid_raw, work, kind="video")
            vid_dur = probe_duration(beat_vid_raw)

            beat_vid = os.path.join(work, f"beat{gi:02d}.mp4")
            # tpad stop_mode=clone holds the last frame if video is short; -t
            # trims if long. Either way the beat video becomes exactly the audio
            # length.
            pad_needed = max(0.0, beat_audio_dur - vid_dur)
            vf = f"tpad=stop_mode=clone:stop_duration={pad_needed:.3f}" \
                if pad_needed > 0.02 else "null"
            _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                  "-i", beat_vid_raw, "-t", f"{beat_audio_dur:.3f}",
                  "-vf", vf, "-r", str(FPS),
                  "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                  "-pix_fmt", PIX, beat_vid], desc=f"beat {gi} conform")

            video_parts.append(beat_vid)
            audio_parts.append(audio_file)

            print(f"    beat {gi:>2} [{group['label']:<14}] "
                  f"{len(segs)} seg  audio {beat_audio_dur:5.2f}s  "
                  f"video {vid_dur:5.2f}s -> conformed")

            # Inter-beat pause: identical silence + black on both tracks, so the
            # two stay locked. Skip after the final beat.
            if gi < len(groups) - 1 and BEAT_PAUSE > 0.01:
                sv = os.path.join(work, f"pause{gi:02d}.mp4")
                sa = os.path.join(work, f"pause{gi:02d}.wav")
                _black(BEAT_PAUSE, sv)
                _silence(BEAT_PAUSE, sa)
                video_parts.append(sv)
                audio_parts.append(sa)

        # Concat all beats + pauses on each track independently. Because every
        # video part was conformed to its audio and both tracks share the exact
        # same pause segments, the two full tracks are the same length.
        full_video = os.path.join(work, "full_video.mp4")
        full_audio = os.path.join(work, "full_audio.wav")
        _concat(video_parts, full_video, work, kind="video")
        _concat(audio_parts, full_audio, work, kind="audio")

        v_dur = probe_duration(full_video)
        a_dur = probe_duration(full_audio)

        # Final mux + polished encode (this is where the GPU encoder is used).
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        enc = pick_encoder()
        _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
              "-i", full_video, "-i", full_audio,
              "-map", "0:v:0", "-map", "1:a:0",
              *enc,
              "-c:a", "aac", "-b:a", "192k",
              "-movflags", "+faststart",
              "-shortest", out_path], desc="final mux")

        out_dur = probe_duration(out_path)
        report = {
            "output": out_path,
            "duration": out_dur,
            "video_track": v_dur,
            "audio_track": a_dur,
            "sync_drift": round(v_dur - a_dur, 3),
            "beats": len(groups),
            "segments": len(segments),
            "size_mb": round(os.path.getsize(out_path) / 1e6, 2),
        }
        return report
    finally:
        if made_temp and not keep_work:
            shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------- CLI

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print("usage:")
        print("  python assemble.py <edl.json> <voiceover_manifest.json> "
              "[--out video.mp4] [--keep-work <dir>]")
        sys.exit(0)

    edl_path, manifest_path = sys.argv[1], sys.argv[2]
    out = "data/explainer.mp4"
    work_dir = None
    keep = False
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    if "--keep-work" in sys.argv:
        work_dir = sys.argv[sys.argv.index("--keep-work") + 1]
        keep = True

    with open(edl_path, encoding="utf-8") as f:
        edl = json.load(f)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    print("=" * 68)
    print("  ASSEMBLE")
    print("=" * 68)
    rep = assemble(edl, manifest, out, work_dir=work_dir, keep_work=keep)
    print("\n" + "=" * 68)
    print(f"  output     : {rep['output']}")
    print(f"  duration   : {rep['duration']:.2f}s "
          f"({rep['size_mb']} MB)")
    print(f"  sync drift : {rep['sync_drift']:+.3f}s "
          f"(video {rep['video_track']:.2f}s vs audio {rep['audio_track']:.2f}s)")
    print(f"  content    : {rep['beats']} beats, {rep['segments']} segments")
    print("=" * 68)
