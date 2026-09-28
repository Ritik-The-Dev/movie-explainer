"""
sourceRead.py — turn a supplied video file into a structured brief.

The script engine can't watch a movie. This module converts one into the two
things it actually needs:

  1. WHAT IS SAID    — a timestamped transcript, so the explainer is built on
                       the real plot rather than an LLM's guess about a title.
  2. WHERE THINGS    — a shot list, so narration can be laid against footage
     HAPPEN            that actually changes when the story does.

It then condenses both into a fixed number of windows — a beat sheet — because
a full 2.5-hour transcript is far too large to hand a language model, and
truncating it arbitrarily loses the third act, which is exactly the part an
explainer video is about.

Also computes per-window visual energy (shot changes per minute), which
hookScore.score_short_hook() consumes to rank Shorts candidates. A great line
over a static frame still loses on Shorts, so this number has to come from
somewhere real.

Input is a file you supply. This module never downloads anything.

Shot detection uses ffmpeg's own scene filter rather than PySceneDetect. That
was a deliberate change after PySceneDetect pulled in opencv-python 5, which
dragged numpy from 1.22 to 2.2 and broke scipy, gruut and Coqui TTS in the
process. ffmpeg was already installed, needs no Python packages at all, found
every cut in a test clip exactly, and did it in 0.15s. Fewer moving parts.

Transcription runs in bounded chunks. This is not an optimisation:
faster-whisper's decode_audio() reads the whole track into RAM as float32, which
is ~1.6 GB for a 7-hour file and raised MemoryError on a real input. Chunking
caps memory at ~77 MB regardless of length, and each chunk is written to disk as
it finishes, so a crash costs one chunk instead of the whole run.

Dependencies (one-time):
    pip install faster-whisper
Usage:
    python src/sourceRead.py --check
    python src/sourceRead.py "src/test_assets/hindi_test_clip.mp4"
    python src/sourceRead.py "<video>" --range 0-22      # minutes
    python src/sourceRead.py "<video>" --force           # ignore caches

--range exists for files holding several episodes. Timecodes stay absolute to
the source, so a brief built from a slice still lines up with the file you hand
to the assembly step.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict

# Windows consoles default to cp1252, which crashes on any transcript
# character outside it (e.g. Korean/Japanese whisper output). Replace
# unprintable chars with ? instead of dying after 30 min of work.
try:
    if sys.stdout is not None:
        sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
WORK_DIR = os.path.join(BASE_DIR, "source_work")

# Flush every line. Loading whisper can take a native library down hard on
# Windows — no Python traceback, no exit message — and anything still sitting in
# a buffer is lost with it, which makes the crash look like it happened a stage
# earlier than it did.
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

# "small" transcribes a 2h film in roughly 30-45 min on CPU and is accurate
# enough for plot comprehension, which is all we need — we are not subtitling.
# Move to "medium" if the briefs come back vague; it roughly triples the time.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")

# ffmpeg's scene filter scores each frame 0-1 against the previous one. 0.30 is
# a good hard-cut threshold; a test clip detected identically at 0.2, 0.3 and
# 0.4, so this is not a delicate number. Lower it only if a film uses lots of
# soft dissolves. (Note: PySceneDetect's equivalent was on a 0-255 scale, so
# don't carry a threshold of 27 over here.)
SCENE_THRESHOLD = float(os.getenv("SCENE_THRESHOLD", "0.30"))

# Frames are downscaled before the scene maths. Verified not to change which
# cuts are found, and it makes the filter far cheaper on 1080p+ sources.
SCENE_SCALE_WIDTH = int(os.getenv("SCENE_SCALE_WIDTH", "320"))

# Target window length for the beat sheet. 4 minutes keeps a 2h film at ~30
# windows, which is a comfortable size for the script engine's context.
WINDOW_SECONDS = float(os.getenv("WINDOW_SECONDS", "240"))
WINDOW_MIN, WINDOW_MAX = 12, 40

# Shot changes per minute that we treat as "maximum energy". Dialogue scenes sit
# around 5-15, action cutting runs 30+.
ENERGY_CEILING_CPM = 30.0

# Dialogue kept per window in the brief. Enough to convey what happens without
# reproducing the screenplay.
DIALOGUE_CHARS_PER_WINDOW = 700

# Transcription is done in chunks, and this is not a tuning knob — it is a
# correctness requirement. faster-whisper's decode_audio() reads the whole track
# into RAM as float32: 16000 samples/s x 4 bytes = 3.84 MB per minute, so a
# 412-minute source needs ~1.6 GB before any working memory, and it raised
# MemoryError on a real file. 20 minutes per chunk caps that at ~77 MB no matter
# how long the input is.
CHUNK_MINUTES = float(os.getenv("WHISPER_CHUNK_MINUTES", "20"))

# Chunks overlap slightly so a sentence spanning a boundary isn't cut in half.
# Segments landing in the overlap are attributed to the earlier chunk and
# dropped from the later one, so nothing is duplicated.
CHUNK_OVERLAP = 3.0

# Language detection reads only this much audio, taken a little way in rather
# than at 00:00 — opening logos are silent and detect as nothing useful. Also
# capped to a fraction of the file, because asking for 120s of a 72s clip made
# the first real test transcribe the whole thing twice.
LANG_PROBE_SECONDS = 120.0
LANG_PROBE_MAX_FRACTION = 0.2

# Skip detection entirely when you already know: --lang hi / --lang en. Saves a
# whole extra pass over the probe slice.
SOURCE_LANG = os.getenv("SOURCE_LANG", "").strip().lower()

# Greedy decoding, not beam search. Measured RTF 2.235 with beam_size=5 on this
# CPU, which puts a 2.5-hour film at over 5 hours — unusable for a daily
# pipeline. We need plot comprehension, not broadcast subtitles, and beam search
# buys accuracy we then throw away when the script engine paraphrases everything
# into Hindi narration anyway. Raise it if briefs come back genuinely confusing.
WHISPER_BEAM = int(os.getenv("WHISPER_BEAM", "1"))

# 0 lets faster-whisper use every core. Set it if transcription needs to share
# the machine with something else.
WHISPER_THREADS = int(os.getenv("WHISPER_THREADS", "0"))

# Speech timing comes from ffmpeg's silencedetect, not from whisper's segment
# timestamps — see detect_speech(). -35 dB over 0.7s is a reasonable "nobody is
# talking" threshold for dialogue mixes.
SPEECH_DB = os.getenv("SPEECH_DB", "-35dB")
SPEECH_MIN_SILENCE = float(os.getenv("SPEECH_MIN_SILENCE", "0.7"))

# Spans shorter than this are discarded. Closing the final span at the file end
# can leave a sliver of a few hundredths of a second, which then shows up as
# 0.1s of "speech" in a window that is genuinely silent — precisely the reading
# we don't want to blur.
MIN_SPEECH_SPAN = 0.15

# Above this, the input is almost certainly several episodes concatenated. We
# still handle it, but a beat sheet spanning a whole season is too coarse to
# write an explainer from, so say so.
MULTI_EPISODE_MINUTES = 150.0


class SourceReadError(RuntimeError):
    pass


# ---------------------------------------------------------------- probing

@dataclass
class MediaInfo:
    path: str
    duration: float
    fps: float
    width: int
    height: int
    has_audio: bool
    # When a --range is in play, duration is the length of the slice and offset
    # is where it starts in the real file. Every timecode we report is absolute,
    # so a beat sheet built from a slice still points at the right moment in the
    # source you actually hand to the assembly step.
    offset: float = 0.0

    def dict(self) -> dict:
        return asdict(self)

    @property
    def stem(self) -> str:
        base = os.path.splitext(os.path.basename(self.path))[0]
        # Keep it short: season filenames are enormous and Windows still has a
        # 260-char path limit by default.
        base = base[:60]
        if self.offset or self.is_slice:
            return f"{base}.{int(self.offset)}-{int(self.offset + self.duration)}"
        return base

    @property
    def is_slice(self) -> bool:
        return bool(getattr(self, "_sliced", False))

    def ff_span(self) -> list:
        """ffmpeg args to select the span, empty if we want the whole file."""
        if not (self.offset or self.is_slice):
            return []
        return ["-ss", f"{self.offset:.3f}", "-t", f"{self.duration:.3f}"]


def apply_range(info: MediaInfo, span: tuple[float, float] | None) -> MediaInfo:
    """Narrow a MediaInfo to a time span given in seconds."""
    if span is None:
        return info
    start, end = span
    start = max(0.0, start)
    end = min(info.duration, end) if end > 0 else info.duration
    if end - start < 1.0:
        raise SourceReadError(
            f"that range is empty — the file is {info.duration / 60:.1f} min "
            f"long and you asked for {start / 60:.1f}-{end / 60:.1f} min")
    sliced = MediaInfo(path=info.path, duration=round(end - start, 3),
                       fps=info.fps, width=info.width, height=info.height,
                       has_audio=info.has_audio, offset=round(start, 3))
    object.__setattr__(sliced, "_sliced", True)
    return sliced


def probe(path: str) -> MediaInfo:
    if not os.path.exists(path):
        raise SourceReadError(f"no such file: {path}")

    def ff(*args: str) -> str:
        out = subprocess.run(["ffprobe", "-v", "error", *args, path],
                             capture_output=True, text=True)
        return out.stdout.strip()

    duration = float(ff("-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1") or 0)
    vstream = ff("-select_streams", "v:0", "-show_entries",
                 "stream=width,height,r_frame_rate",
                 "-of", "csv=s=,:p=0").split(",")
    astream = ff("-select_streams", "a:0", "-show_entries",
                 "stream=index", "-of", "default=nw=1:nk=1")

    width = int(vstream[0]) if len(vstream) > 0 and vstream[0].isdigit() else 0
    height = int(vstream[1]) if len(vstream) > 1 and vstream[1].isdigit() else 0
    fps = 0.0
    if len(vstream) > 2 and "/" in vstream[2]:
        num, den = vstream[2].split("/")[:2]
        fps = round(float(num) / float(den), 3) if float(den) else 0.0

    if duration <= 0:
        raise SourceReadError(f"could not read a duration from {path} — "
                              f"is it a video file?")

    return MediaInfo(path=path, duration=duration, fps=fps, width=width,
                     height=height, has_audio=bool(astream))


# ---------------------------------------------------------------- audio

def extract_audio(info: MediaInfo, *, force: bool = False) -> str:
    """16 kHz mono wav — what whisper wants, and ~100x smaller than the video."""
    if not info.has_audio:
        raise SourceReadError("this file has no audio track, so there is "
                              "nothing to transcribe")
    os.makedirs(WORK_DIR, exist_ok=True)
    dst = os.path.join(WORK_DIR, f"{info.stem}.16k.wav")

    if os.path.exists(dst) and not force and os.path.getsize(dst) > 1024:
        print(f"  audio  : cached ({os.path.getsize(dst) / 1e6:.0f} MB)")
        return dst

    print(f"  audio  : extracting {info.duration / 60:.0f} min to 16 kHz mono")
    t0 = time.time()
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         *info.ff_span(), "-i", info.path, "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", dst],
        check=True)
    print(f"           done in {time.time() - t0:.0f}s")
    return dst


# ---------------------------------------------------------------- transcript

def _slice_wav(src: str, start: float, dur: float, dst: str) -> str:
    """Cut a span out of the 16 kHz wav. Cheap — no re-encoding of the video."""
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", src,
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst],
        check=True)
    return dst


def _load_model():
    # Announce before importing, not after. The import pulls in ctranslate2's
    # native libraries, which can abort the process outright on Windows with no
    # traceback at all — so if the last thing you see is this line, the failure
    # is in the library load, not in anything above it.
    print(f"  script : importing faster_whisper")
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise SourceReadError(
            "faster-whisper is not installed. Run:\n"
            "  pip install faster-whisper --break-system-packages"
        ) from e

    print(f"  script : loading whisper '{WHISPER_MODEL}' "
          f"({WHISPER_COMPUTE}, CPU, beam={WHISPER_BEAM})")
    t0 = time.time()
    # cpu_threads is only passed when explicitly set. Passing 0 is documented as
    # "use the default", but it is still a different code path inside
    # ctranslate2 than omitting the argument, and this is the one thing that
    # changed between a run that worked and a run that died silently. Not worth
    # the risk for a knob nobody needs by default.
    kwargs = {}
    if WHISPER_THREADS > 0:
        kwargs["cpu_threads"] = WHISPER_THREADS
        print(f"           limiting to {WHISPER_THREADS} CPU threads")
    model = WhisperModel(WHISPER_MODEL, device="cpu",
                         compute_type=WHISPER_COMPUTE, **kwargs)
    print(f"           model loaded in {time.time() - t0:.0f}s")
    return model


def detect_language(model, wav: str, info: MediaInfo) -> tuple[str, float]:
    """Identify the language from a short slice taken a little way in.

    Three reasons this reads a slice rather than the file. Memory:
    decode_audio() loads the entire track as float32 and died with MemoryError
    on a 412-minute input. Accuracy: audio at 00:00 is usually a silent
    distributor logo, which detects as noise. Speed: the slice is also capped to
    a fraction of the source, because on a short clip an uncapped 120s probe
    means transcribing the whole file an extra time.
    """
    if SOURCE_LANG:
        print(f"           language forced to {SOURCE_LANG} "
              f"(detection skipped)")
        return SOURCE_LANG, 1.0

    os.makedirs(WORK_DIR, exist_ok=True)
    start = min(info.duration * 0.10, 600.0)
    dur = min(LANG_PROBE_SECONDS, info.duration * LANG_PROBE_MAX_FRACTION)
    probe_wav = os.path.join(WORK_DIR, "_langprobe.wav")
    try:
        _slice_wav(wav, start, dur, probe_wav)
        t0 = time.time()
        _, meta = model.transcribe(probe_wav, vad_filter=True,
                                   without_timestamps=True, beam_size=1)
        lang, prob = meta.language, meta.language_probability
        print(f"           detected {lang} (p={prob:.2f}) from a "
              f"{dur:.0f}s probe in {time.time() - t0:.0f}s")
        return lang, prob
    finally:
        if os.path.exists(probe_wav):
            os.remove(probe_wav)


def transcribe(wav: str, info: MediaInfo, *, force: bool = False) -> dict:
    """Timestamped transcript, chunked and resumable.

    Chunking is what makes this survive long inputs at all (see CHUNK_MINUTES).
    Each chunk's result is written to disk as it completes, so a crash or a
    Ctrl-C two hours into a film costs one chunk rather than the whole run.
    """
    stem = info.stem
    cache = os.path.join(WORK_DIR, f"{stem}.transcript.json")
    if os.path.exists(cache) and not force:
        with open(cache, encoding="utf-8") as f:
            data = json.load(f)
        print(f"  script : cached transcript, {len(data['segments'])} segments "
              f"({data.get('language')})")
        return data

    os.makedirs(WORK_DIR, exist_ok=True)
    model = _load_model()

    detected, confidence = detect_language(model, wav, info)
    # Hindi or English we keep verbatim. Anything else we translate to English,
    # because the script engine reasons far better over English than over, say,
    # romanised Korean — and this costs one pass, not two.
    task = "transcribe" if detected in ("hi", "en") else "translate"
    if task == "translate":
        print(f"           source is {detected}, translating to English "
              f"for the script engine")

    chunk_len = CHUNK_MINUTES * 60
    n_chunks = max(1, math.ceil(info.duration / chunk_len))
    print(f"           transcribing {info.duration / 60:.0f} min in "
          f"{n_chunks} chunk(s) of {CHUNK_MINUTES:.0f} min "
          f"(VAD on — silence and score are skipped)")

    slice_path = os.path.join(WORK_DIR, f"{stem}._chunk.wav")
    out: list[dict] = []
    t0 = time.time()

    for ci in range(n_chunks):
        part = os.path.join(WORK_DIR, f"{stem}.part{ci:03d}.json")
        # Two clocks: wav_off indexes into the extracted wav, abs_off is the
        # position in the original file. Everything we emit uses abs_off, so a
        # brief built from --range still points at real source timecodes.
        wav_off = ci * chunk_len
        abs_off = info.offset + wav_off

        if os.path.exists(part) and not force:
            with open(part, encoding="utf-8") as f:
                segs = json.load(f)
            print(f"           chunk {ci + 1}/{n_chunks}  cached "
                  f"({len(segs)} segments)")
            out.extend(segs)
            continue

        dur = min(chunk_len + CHUNK_OVERLAP, info.duration - wav_off)
        if dur <= 0.1:
            continue
        _slice_wav(wav, wav_off, dur, slice_path)

        segments, _ = model.transcribe(
            slice_path,
            task=task,
            vad_filter=True,                   # big win: films are mostly not speech
            vad_parameters={"min_silence_duration_ms": 700},
            beam_size=WHISPER_BEAM,
            condition_on_previous_text=False,  # stops runaway repetition loops
        )

        # Timestamps come back relative to the slice, so shift them. Anything
        # starting past the nominal chunk end belongs to the next chunk, which
        # will cover it properly — drop it here so nothing is counted twice.
        limit = abs_off + chunk_len
        segs = []
        for s in segments:
            abs_start = s.start + abs_off
            if abs_start >= limit and ci < n_chunks - 1:
                continue
            text = s.text.strip()
            if text:
                segs.append({"start": round(abs_start, 2),
                             "end": round(s.end + abs_off, 2),
                             "text": text})

        with open(part, "w", encoding="utf-8") as f:
            json.dump(segs, f, indent=2, ensure_ascii=False)
        out.extend(segs)

        done_s = min(wav_off + chunk_len, info.duration)
        elapsed = time.time() - t0
        eta = (elapsed / done_s) * (info.duration - done_s) if done_s else 0
        print(f"           chunk {ci + 1}/{n_chunks}  "
              f"{100 * done_s / info.duration:5.1f}%  "
              f"{len(segs):>4} segments  "
              f"{elapsed / 60:.1f} min elapsed, ~{eta / 60:.0f} min left")

    if os.path.exists(slice_path):
        os.remove(slice_path)

    out.sort(key=lambda s: s["start"])
    wall = time.time() - t0
    data = {
        "language": detected,
        "language_probability": round(confidence, 3),
        "task": task,
        "model": WHISPER_MODEL,
        "chunks": n_chunks,
        "segments": out,
        "wall_seconds": round(wall, 1),
        "rtf": round(wall / info.duration, 3) if info.duration else 0,
    }
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Per-chunk files exist only to survive a crash. The merged cache is
    # authoritative now, so clear them rather than leave stale copies around.
    for ci in range(n_chunks):
        part = os.path.join(WORK_DIR, f"{stem}.part{ci:03d}.json")
        if os.path.exists(part):
            os.remove(part)

    spoken = sum(s["end"] - s["start"] for s in out)
    print(f"           {len(out)} segments, {spoken / 60:.1f} min of speech "
          f"in a {info.duration / 60:.0f} min source")
    print(f"           took {wall / 60:.1f} min (RTF {data['rtf']})")
    return data


# ---------------------------------------------------------------- speech timing

def detect_speech(wav: str, info: MediaInfo, *, force: bool = False) -> list:
    """Where there is actually sound, measured from the audio.

    Whisper's segment timestamps cannot be used for this. With vad_filter on it
    strips silence, transcribes the concatenated speech, then maps timestamps
    back over the original span — and the segment boundaries get redistributed
    across the removed gaps. Measured on the test clip: 37.0s of planted speech
    in three bursts was reported as 52.1s, a 41% overstatement, with one segment
    spanning an 8-second silence it had no business covering.

    ffmpeg's silencedetect measures the waveform instead, costs about a second
    for a whole film, and needs no Python packages.

    Caveat worth knowing: on a scored film almost nothing is below -35 dB,
    because music and room tone never stop. So this reliably finds "no audio at
    all" but does not separate dialogue from soundtrack. For that, use the
    dialogue character count, which is what ranks the beat sheet.
    """
    stem = info.stem
    cache = os.path.join(WORK_DIR, f"{stem}.speech.json")
    if os.path.exists(cache) and not force:
        with open(cache, encoding="utf-8") as f:
            spans = json.load(f)
        print(f"  speech : cached, {len(spans)} spans")
        return spans

    print(f"  speech : measuring audible spans "
          f"({SPEECH_DB} over {SPEECH_MIN_SILENCE}s)")
    t0 = time.time()
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", wav,
         "-af", f"silencedetect=noise={SPEECH_DB}:d={SPEECH_MIN_SILENCE}",
         "-f", "null", "-"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace")

    # silencedetect reports the gaps; we want their complement.
    silences: list[list] = []
    for line in (proc.stderr or "").splitlines():
        if "silence_start:" in line:
            try:
                silences.append([float(line.split("silence_start:")[1].split()[0]),
                                 None])
            except (IndexError, ValueError):
                continue
        elif "silence_end:" in line and silences and silences[-1][1] is None:
            try:
                silences[-1][1] = float(line.split("silence_end:")[1].split()[0])
            except (IndexError, ValueError):
                silences[-1][1] = info.duration

    # An unterminated silence runs to the end of the file.
    for s in silences:
        if s[1] is None:
            s[1] = info.duration

    spans: list[dict] = []
    cursor = 0.0
    for s_start, s_end in silences:
        if s_start - cursor >= MIN_SPEECH_SPAN:
            spans.append({"start": round(info.offset + cursor, 2),
                          "end": round(info.offset + s_start, 2)})
        cursor = max(cursor, s_end)
    if info.duration - cursor >= MIN_SPEECH_SPAN:
        spans.append({"start": round(info.offset + cursor, 2),
                      "end": round(info.offset + info.duration, 2)})

    os.makedirs(WORK_DIR, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(spans, f, indent=2)

    audible = sum(s["end"] - s["start"] for s in spans)
    print(f"           {len(spans)} audible spans, {audible / 60:.1f} min of "
          f"{info.duration / 60:.1f} ({100 * audible / info.duration:.0f}%), "
          f"took {time.time() - t0:.0f}s")
    return spans


def _overlap(spans: list, start: float, end: float) -> float:
    return sum(max(0.0, min(s["end"], end) - max(s["start"], start))
               for s in spans)


# ---------------------------------------------------------------- shots

def detect_shots(info: MediaInfo, *, force: bool = False) -> list:
    """Shot boundary list via ffmpeg's scene filter. Cached.

    No Python dependencies: ffmpeg scores each frame against its predecessor and
    we keep the frames that exceed the threshold. Those are the cut points.
    """
    stem = info.stem
    cache = os.path.join(WORK_DIR, f"{stem}.shots.json")
    if os.path.exists(cache) and not force:
        with open(cache, encoding="utf-8") as f:
            shots = json.load(f)
        print(f"  shots  : cached, {len(shots)} shots")
        return shots

    print(f"  shots  : scanning for cuts (threshold {SCENE_THRESHOLD}, "
          f"downscaled to {SCENE_SCALE_WIDTH}px)")
    t0 = time.time()

    vf = (f"scale={SCENE_SCALE_WIDTH}:-2,"
          f"select='gt(scene,{SCENE_THRESHOLD})',showinfo")
    proc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-nostats", *info.ff_span(),
         "-i", info.path, "-filter:v", vf, "-an", "-f", "null", "-"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace")

    cuts: list[float] = []
    last_report = 0.0
    assert proc.stderr is not None
    for line in proc.stderr:
        if "pts_time:" not in line:
            continue
        try:
            frag = line.split("pts_time:")[1].split()[0]
            t = float(frag)
        except (IndexError, ValueError):
            continue
        cuts.append(t)
        # Progress matters here: a 2h film means minutes of silence otherwise.
        if t - last_report > 600:
            print(f"           {100 * t / info.duration:5.1f}%  "
                  f"{t / 60:6.1f} min  {len(cuts)} cuts  "
                  f"({time.time() - t0:.0f}s)")
            last_report = t
    proc.wait()

    if proc.returncode != 0 and not cuts:
        raise SourceReadError(
            f"ffmpeg failed to scan {info.path} for scene cuts "
            f"(exit {proc.returncode})")

    # Turn cut points into spans, in absolute source time. Guard against a cut
    # at the very start producing a zero-length shot, and always close the final
    # shot at the real end of the analysed span.
    lo, hi = info.offset, info.offset + info.duration
    bounds = [lo] + [c + info.offset for c in cuts if c > 0.05] + [hi]
    bounds = sorted(set(round(b, 3) for b in bounds if lo <= round(b, 3) <= hi))
    shots = [{"i": i, "start": bounds[i], "end": bounds[i + 1],
              "duration": round(bounds[i + 1] - bounds[i], 3)}
             for i in range(len(bounds) - 1)]

    os.makedirs(WORK_DIR, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(shots, f, indent=2)

    avg = (sum(s["duration"] for s in shots) / len(shots)) if shots else 0
    print(f"           {len(shots)} shots, mean {avg:.1f}s, "
          f"took {time.time() - t0:.0f}s")
    return shots


# ---------------------------------------------------------------- beat sheet

def _tc(t: float) -> str:
    """h:mm:ss, dropping the hour for short sources. Season files run past
    99 minutes, where a bare mm:ss stops being readable."""
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def build_windows(info: MediaInfo, transcript: dict, shots: list,
                  speech: list | None = None) -> list:
    """Condense the film into a fixed number of windows for the script engine.

    Fixed count, not fixed length, so the brief stays a predictable size whether
    the input is a 40-minute episode or a 3-hour film.
    """
    n = int(round(info.duration / WINDOW_SECONDS))
    n = max(WINDOW_MIN, min(WINDOW_MAX, n))
    span = info.duration / n

    windows: list[dict] = []
    segments = transcript.get("segments") or []

    for i in range(n):
        start = info.offset + i * span
        end = info.offset + (i + 1) * span

        lines = [s["text"] for s in segments
                 if s["start"] < end and s["end"] > start and s["text"]]
        full_dialogue = " ".join(lines).strip()
        dialogue = full_dialogue
        truncated = len(dialogue) > DIALOGUE_CHARS_PER_WINDOW
        if truncated:
            dialogue = dialogue[:DIALOGUE_CHARS_PER_WINDOW].rsplit(" ", 1)[0] + " …"

        in_window = [s for s in shots if s["start"] < end and s["end"] > start]
        cpm = len(in_window) / (span / 60) if span else 0
        energy = round(min(1.0, cpm / ENERGY_CEILING_CPM), 3)

        w = {
            "i": i,
            "start": round(start, 1),
            "end": round(end, 1),
            "timecode": f"{_tc(start)}-{_tc(end)}",
            "shot_count": len(in_window),
            "shots_per_min": round(cpm, 1),
            "energy": energy,          # feeds hookScore.score_short_hook
            # Density of what is said, in characters. This is the reliable
            # dialogue signal — it doesn't depend on timestamp precision at all,
            # which matters because whisper's are smeared (see detect_speech).
            "dialogue_chars": len(full_dialogue),
            "dialogue": dialogue,
            "dialogue_truncated": truncated,
        }

        if speech is not None:
            w["speech_seconds"] = round(_overlap(speech, start, end), 1)
            w["speech_source"] = "silencedetect"
        else:
            # Fallback only. Overstates, because whisper spreads segment
            # boundaries across the silence its own VAD removed.
            w["speech_seconds"] = round(sum(
                min(s["end"], end) - max(s["start"], start)
                for s in segments
                if s["start"] < end and s["end"] > start), 1)
            w["speech_source"] = "whisper (approximate)"

        windows.append(w)

    return windows


def build_brief(path: str, *, force: bool = False,
                span: tuple[float, float] | None = None) -> dict:
    """Full pipeline: probe, extract, transcribe, detect shots, condense."""
    print("=" * 68)
    print(f"  SOURCE READ  {os.path.basename(path)}")
    print("=" * 68)

    full = probe(path)
    info = apply_range(full, span)
    print(f"  media  : {full.duration / 60:.1f} min, {full.width}x{full.height}, "
          f"{full.fps} fps, audio={full.has_audio}")
    if info is not full:
        print(f"  range  : {_tc(info.offset)}-{_tc(info.offset + info.duration)} "
              f"({info.duration / 60:.1f} min of {full.duration / 60:.1f})")
    elif full.duration / 60 > MULTI_EPISODE_MINUTES:
        # A 400-minute file is a season, not an episode. We'll process it, but
        # 40 windows across 7 hours is ~10 min per window, which is far too
        # coarse to write beats from — and it's ~2.5 hours of CPU to find out.
        print(f"\n  NOTE   : {full.duration / 60:.0f} min is long enough that "
              f"this is probably several episodes in one file.")
        print(f"           A brief spanning all of it averages "
              f"{full.duration / 60 / WINDOW_MAX:.0f} min per window, which is "
              f"too coarse to write an explainer from.")
        print(f"           Consider one episode at a time:")
        print(f"             python src/sourceRead.py \"{os.path.basename(path)}\" "
              f"--range 0-22")
        print(f"           Ranges are in minutes and every timecode stays "
              f"absolute to the source file.\n")

    # Measured on this machine, not guessed: whisper-small/int8 came in at
    # RTF 2.235 with beam_size=5. Greedy decoding is roughly 2-3x faster, so
    # ~0.9 is the working figure. Overshooting an estimate is far kinder than
    # promising 40 minutes and taking three hours.
    rtf_guess = 0.9 if WHISPER_BEAM <= 1 else 2.3
    est_min = info.duration / 60 * rtf_guess
    print(f"  budget : expect roughly {est_min:.0f} min for transcription "
          f"(RTF ~{rtf_guess} at beam={WHISPER_BEAM}) plus a couple for "
          f"shots and speech")

    wav = extract_audio(info, force=force)
    transcript = transcribe(wav, info, force=force)
    speech = detect_speech(wav, info, force=force)
    shots = detect_shots(info, force=force)
    windows = build_windows(info, transcript, shots, speech)

    brief = {
        "source": info.dict(),
        "source_duration": full.duration,
        "range": [info.offset, round(info.offset + info.duration, 3)],
        "language": transcript.get("language"),
        "task": transcript.get("task"),
        "whisper_model": transcript.get("model"),
        "whisper_beam": WHISPER_BEAM,
        "shot_count": len(shots),
        "speech_span_count": len(speech),
        "window_count": len(windows),
        "windows": windows,
        "transcript_segments": len(transcript.get("segments") or []),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    dst = os.path.join(DATA_DIR, f"source_brief_{info.stem}.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(brief, f, indent=2, ensure_ascii=False)

    print(f"\n  brief  : {dst}")
    print(f"           {len(windows)} windows, {len(shots)} shots, "
          f"{brief['transcript_segments']} transcript segments")
    print("=" * 68)
    return brief


# ---------------------------------------------------------------- reporting

def print_brief(brief: dict, *, chars: int = 96) -> None:
    print(f"\n  {'win':>3} {'timecode':<17} {'cuts':>5} {'cpm':>6} "
          f"{'energy':>7} {'talk':>6} {'chars':>6}  dialogue")
    print("  " + "-" * 114)
    for w in brief["windows"]:
        bar = "#" * int(w["energy"] * 10)
        # One decimal, not zero: a 0.3s overlap rounding to "0s" is
        # indistinguishable from genuine silence, and genuine silence in a
        # dialogue window is the failure we most want to notice.
        print(f"  {w['i']:>3} {w['timecode']:<17} {w['shot_count']:>5} "
              f"{w['shots_per_min']:>6.1f} {bar:<7} "
              f"{w['speech_seconds']:>5.1f}s {w.get('dialogue_chars', 0):>6}  "
              f"{w['dialogue'][:chars]}")

    energetic = sorted(brief["windows"], key=lambda w: -w["energy"])[:5]
    # Ranked on characters spoken, not seconds. Seconds of audible sound says
    # little on a scored film, where the music never stops; characters of
    # transcript is a direct measure of how much is actually being said.
    talky = sorted(brief["windows"],
                   key=lambda w: -w.get("dialogue_chars", 0))[:5]
    print(f"\n  most visually active : "
          f"{', '.join(w['timecode'] for w in energetic)}")
    print(f"  most dialogue-heavy  : "
          f"{', '.join(w['timecode'] for w in talky)}")

    # Loud and wordless is the ideal Shorts window: something is happening on
    # screen and the narration has room to explain it.
    quiet_action = sorted(brief["windows"],
                          key=lambda w: -(w["energy"] * 1000
                                          - w.get("dialogue_chars", 0)))[:3]
    print(f"  action, little said  : "
          f"{', '.join(w['timecode'] for w in quiet_action)}")

    src = (brief["windows"][0].get("speech_source", "?")
           if brief["windows"] else "?")
    print(f"\n  talk column measured by: {src}")
    print("  Visually active windows are where Shorts footage should come from.")
    print("  Dialogue-heavy windows are where the plot actually turns.")


def check_deps(*, load_model: bool = False) -> None:
    print("dependency check\n")
    ok = True

    # shutil.which is cross-platform. My first version shelled out to `which`,
    # which doesn't exist on Windows and crashed the whole check.
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        print(f"  {tool:<18} {'OK  ' + path if path else 'MISSING'}")
        ok = ok and bool(path)

    try:
        import faster_whisper
        print(f"  {'faster_whisper':<18} OK  "
              f"{getattr(faster_whisper, '__version__', '?')}")
    except ImportError:
        print(f"  {'faster_whisper':<18} MISSING  ->  pip install faster-whisper")
        ok = False

    for mod in ("ctranslate2", "numpy"):
        try:
            got = __import__(mod)
            print(f"  {mod:<18} OK  {getattr(got, '__version__', '?')}")
        except ImportError:
            print(f"  {mod:<18} MISSING")

    # Deliberately not required: shot detection is pure ffmpeg now.
    print(f"  {'scenedetect':<18} not needed (ffmpeg does this)")
    print(f"  {'opencv':<18} not needed (ffmpeg does this)")

    if ok and load_model:
        # The one step that can kill the process without raising anything
        # Python can catch. Isolating it here means a silent death is
        # unambiguous rather than looking like a bug in the pipeline.
        print(f"\nloading the model, which is the step most likely to fail hard")
        try:
            _load_model()
            print("model loads fine")
        except SourceReadError as e:
            print(f"FAILED: {e}")
            ok = False

    print("\n" + ("all good — point me at a real video file"
                  if ok else "install what's missing above, then re-run"))


def parse_range(args: list) -> tuple[float, float] | None:
    """--range 0-22  (minutes). Also accepts 'START-' to mean 'to the end'."""
    if "--range" not in args:
        return None
    i = args.index("--range")
    if i + 1 >= len(args):
        raise SourceReadError("--range needs a value, e.g. --range 0-22")
    raw = args[i + 1].strip()
    if "-" not in raw:
        raise SourceReadError(
            f"--range wants START-END in minutes, e.g. --range 0-22 "
            f"(got {raw!r})")
    a, b = raw.split("-", 1)
    try:
        start = float(a) * 60 if a.strip() else 0.0
        end = float(b) * 60 if b.strip() else 0.0     # 0 means 'to the end'
    except ValueError as e:
        raise SourceReadError(
            f"--range values must be numbers in minutes (got {raw!r})") from e
    if end and end <= start:
        raise SourceReadError(
            f"--range end must be after start (got {raw!r})")
    return start, end


# ---------------------------------------------------------------- CLI

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        print("usage:")
        print("  python src/sourceRead.py --check")
        print("  python src/sourceRead.py --check --model          "
              "# also try loading whisper, which can crash hard")
        print("  python src/sourceRead.py \"src/test_assets/hindi_test_clip.mp4\"")
        print("  python src/sourceRead.py \"<video>\" --range 0-22   "
              "# minutes, for one episode of a season file")
        print("  python src/sourceRead.py \"<video>\" --lang hi      "
              "# skip language detection")
        print("  python src/sourceRead.py \"<video>\" --force        "
              "# ignore caches")
        sys.exit(0)

    if args[0] == "--check":
        check_deps(load_model="--model" in args)
        sys.exit(0)

    # --lang skips the detection pass entirely. Set before anything reads it.
    if "--lang" in args:
        i = args.index("--lang")
        if i + 1 >= len(args):
            print("\nERROR: --lang needs a value, e.g. --lang hi")
            sys.exit(1)
        globals()["SOURCE_LANG"] = args[i + 1].strip().lower()

    try:
        b = build_brief(args[0], force="--force" in args,
                        span=parse_range(args))
        print_brief(b)
    except SourceReadError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    except MemoryError:
        print("\nERROR: ran out of memory.")
        print(f"Transcription already works in {CHUNK_MINUTES:.0f}-minute "
              f"chunks, so if this still happens, lower it:")
        print("  set WHISPER_CHUNK_MINUTES=10")
        sys.exit(1)
