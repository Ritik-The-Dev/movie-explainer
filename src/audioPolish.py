"""
audioPolish.py — make TTS output stop sounding like TTS.

Measured on a real chatterbox Hindi sample, two artifacts explain most of the
"not quite human" feeling, and neither is a model problem:

  1. SPECTRALLY DARK. 99% of energy below ~1.5 kHz. Speech needs presence
     (2-5 kHz, intelligibility and consonant definition) and air (8 kHz+,
     breath and sibilance). Without them the voice sounds muffled and distant.

  2. UNIFORM PAUSES. Every pause landed in a 0.18-0.38s band. Real narration
     varies pause length by punctuation and dramatic intent. Rhythmic sameness
     reads as synthetic even when the pitch contour is expressive.

This module fixes (1) with an EQ/dynamics chain, and gives the voiceover module
the tools to fix (2) by generating per-line and inserting authored silence.

Requires ffmpeg on PATH. No API keys, no network.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile

# ---------------------------------------------------------------- the chain

# Each stage exists for a measured reason, not because it's conventional.
FILTERS = [
    # rumble below speech: sample measured 1.48% of energy under 80 Hz
    "highpass=f=75",
    # de-mud: scoop the boxiness that makes cloned voices sound like a phone call
    "equalizer=f=250:t=q:w=1.1:g=-2.5",
    # presence: the band that was missing. consonant definition lives here
    "equalizer=f=3000:t=q:w=0.9:g=3.5",
    # air: breath and sibilance, the strongest "human" cue in the spectrum
    "highshelf=f=8000:g=2.5",
    # gentle glue so loud syllables don't jump out after the EQ lift
    "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=180:makeup=1.5",
    # de-ess: the presence+air lift can make /s/ and /sh/ spit
    "equalizer=f=6500:t=q:w=1.4:g=-1.5",
    # YouTube normalises toward -14 LUFS; -16 for speech leaves headroom
    "loudnorm=I=-16:TP=-1.5:LRA=11",
]


def polish(src: str, dst: str, *, extra: list | None = None) -> str:
    """Apply the chain. Returns dst."""
    chain = ",".join(FILTERS + (extra or []))
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", src, "-af", chain, "-ar", "24000", "-ac", "1", dst]
    subprocess.run(cmd, check=True)
    return dst


# ---------------------------------------------------------------- pause map

# How long to breathe after each kind of boundary. These are the numbers that
# break the uniform-pause tell — note the spread, 0.18 to 0.85, versus the
# 0.18-0.38 band the raw model produced.
PAUSE_AFTER = {
    "comma":      0.20,   # ,  ।-less clause break
    "clause":     0.30,   # — or ;
    "sentence":   0.48,   # । or . or ?
    "beat":       0.70,   # authored dramatic pause
    "reveal":     0.85,   # before the payoff line
    "paragraph":  0.95,   # scene change
}


def split_for_speech(text: str) -> list:
    """Break narration into speakable units with a pause class for each.

    Generating per unit and stitching with authored silence beats handing the
    model a paragraph, for two reasons: the model's own pause logic is flat,
    and long inputs make prosody drift by the end of the chunk.
    """
    units: list = []
    # split on Devanagari danda and western sentence enders, keeping them
    parts = re.split(r"(?<=[।?!\.])\s+", text.strip())
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        # further split very long sentences at commas so prosody stays tight
        if len(part) > 120 and "," in part:
            subs = [s.strip() for s in part.split(",") if s.strip()]
            for j, s in enumerate(subs):
                last = j == len(subs) - 1
                units.append({
                    "text": s + ("," if not last else ""),
                    "pause": PAUSE_AFTER["sentence"] if last else PAUSE_AFTER["comma"],
                })
        else:
            units.append({"text": part, "pause": PAUSE_AFTER["sentence"]})

    if units:
        units[-1]["pause"] = PAUSE_AFTER["paragraph"]
    return units


def apply_beat_hints(units: list, *, beat: str = "") -> list:
    """Lengthen the pause before a payoff so the reveal lands.

    A REVEAL beat read at conversational pace is the most common way an
    otherwise good explainer script falls flat.
    """
    if not units:
        return units
    if beat.upper() in ("REVEAL", "TWIST", "CLIMAX"):
        # breathe before the last line, and let it hang afterwards
        if len(units) >= 2:
            units[-2]["pause"] = PAUSE_AFTER["reveal"]
        units[-1]["pause"] = PAUSE_AFTER["paragraph"]
    elif beat.upper() in ("HOOK", "COLD_OPEN"):
        # hooks want urgency: tighten everything
        for u in units:
            u["pause"] = min(u["pause"], PAUSE_AFTER["comma"])
    return units


# ---------------------------------------------------------------- instruct

def build_instruct(meta: dict) -> str:
    """Turn script_writer's voiceMetadata into a Voicebox `instruct` string.

    Voicebox caps instruct at 500 chars. The existing script engine already
    emits emotion / energy / pace / emphasis / pause_after_words per scene, so
    this is the highest-leverage realism win available: the direction is
    already being written, it just wasn't reaching the voice.
    """
    bits: list[str] = []

    emotion = (meta.get("emotion") or "").strip()
    if emotion:
        bits.append(f"Speak with a {emotion} tone")

    energy = meta.get("energy")
    if isinstance(energy, (int, float)):
        if energy >= 0.8:
            bits.append("high energy, leaning forward, urgent")
        elif energy >= 0.55:
            bits.append("engaged and warm, conversational")
        elif energy >= 0.3:
            bits.append("calm and measured")
        else:
            bits.append("quiet, almost confiding")

    pace = (meta.get("pace") or "").strip().lower()
    if pace == "fast":
        bits.append("brisk pace, clipped")
    elif pace == "slow":
        bits.append("unhurried, let words land")
    elif pace:
        bits.append("natural pace")

    emph = [str(w) for w in (meta.get("emphasis") or [])][:4]
    if emph:
        bits.append("stress these words: " + ", ".join(emph))

    pauses = [str(w) for w in (meta.get("pause_after_words") or [])][:3]
    if pauses:
        bits.append("pause briefly after: " + ", ".join(pauses))

    bits.append("Sound like a friend telling a story, not a newsreader")

    out = ". ".join(bits) + "."
    return out[:500]


# ---------------------------------------------------------------- stitching

def stitch(segments: list, dst: str, *, sample_rate: int = 24000) -> str:
    """Concatenate audio files with authored silence between them.

    segments: [{"file": path, "pause": seconds_after}, ...]
    """
    if not segments:
        raise ValueError("nothing to stitch")

    work = tempfile.mkdtemp(prefix="stitch_")
    try:
        listing = os.path.join(work, "concat.txt")
        pieces: list[str] = []

        for i, seg in enumerate(segments):
            pieces.append(seg["file"])
            pause = float(seg.get("pause") or 0)
            if pause > 0.01 and i < len(segments) - 1:
                sil = os.path.join(work, f"sil_{i}.wav")
                subprocess.run(
                    ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                     "-f", "lavfi", "-i",
                     f"anullsrc=r={sample_rate}:cl=mono",
                     "-t", f"{pause:.3f}", sil],
                    check=True)
                pieces.append(sil)

        with open(listing, "w", encoding="utf-8") as f:
            for p in pieces:
                f.write(f"file '{os.path.abspath(p)}'\n")

        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "concat", "-safe", "0", "-i", listing,
             "-ar", str(sample_rate), "-ac", "1", dst],
            check=True)
        return dst
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ------------------------------------------------------- garbage trimming

# Chatterbox renders two kinds of junk around the words, measured 27 Sep 2026
# on real Hindi units: (1) harsh broadband bursts (flat spectrum, ~-29 dB)
# and (2) low hums that SWELL after a lull (a real echo tail decays; these
# rise from -70 dB to -27 dB, then cut dead). Both emerge from near-silence
# after speech has ended — that "re-appearance from quiet" is the signature
# real speech never has, so the trim keys on it instead of raw energy.
_ANCHOR_RMS_DB = -32.0   # confident-speech energy floor
_ANCHOR_FLAT = 0.50      # confident speech is tone-like; hiss is flat
_WIN_S = 0.02            # analysis window (20 ms)


def _frame_stats(x, sr):
    """Per-window RMS (dBFS) and spectral flatness (0=tonal, 1=white)."""
    import numpy as np
    win = int(sr * _WIN_S)
    m = len(x) // win
    fr = x[:m * win].reshape(m, win)
    rms = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-9)
    spec = np.abs(np.fft.rfft(fr, axis=1)) + 1e-9
    flat = np.exp(np.log(spec).mean(axis=1)) / spec.mean(axis=1)
    return rms, flat, win, m


def _emerged_from_quiet(pos, rms):
    """True when the 0.4 s before `pos` bottomed out below -50 dB."""
    import numpy as np
    lo = max(0, pos - int(0.40 / _WIN_S))
    return pos > lo and float(np.min(rms[lo:pos])) < -50.0


def _swell_spans(rms):
    """Spans [u, v] of swelling-hum model garbage.

    Signature, measured on real units: emerges from deep quiet (the 0.5 s
    before sits below -52 dB), then climbs >= 12 dB to an audible peak at
    >= +15 dB/s, smoothly (small linear-fit residual — syllables never look
    like this), over >= 0.3 s — and then dies: a real phrase peak is
    followed by more speech, the swell by silence. An echo tail always
    decays, so this never matches one; real onsets peak within ~0.1 s, so
    the slowness requirement keeps them safe.
    """
    import numpy as np
    W = _WIN_S
    n = len(rms)
    lookback = int(0.50 / W)
    horizon = int(1.00 / W)
    min_len = int(0.30 / W)
    spans = []
    u = 0
    while u < n - min_len:
        if rms[u] < -48.0:
            lo = max(0, u - lookback)
            if u > lo and float(np.min(rms[lo:u])) < -52.0:
                seg = rms[u:u + horizon]
                v = u + int(np.argmax(seg))
                if v - u >= min_len and seg[v - u] > -40.0:
                    # committed rise start (skip the lull bottom)
                    u2 = u
                    for i in range(u, v):
                        if rms[i] > -55.0 and np.all(rms[i:i + 5] > -55.0):
                            u2 = i
                            break
                    y = rms[u2:v + 1]
                    if len(y) >= min_len:
                        t = np.arange(len(y)) * W
                        coef = np.polyfit(t, y, 1)
                        resid = y - np.polyval(coef, t)
                        rise = float(y[-1] - y[0])
                        if (coef[0] > 15.0 and float(resid.std()) < 5.0
                                and rise >= 12.0):
                            tail = rms[v + 8:]
                            if len(tail) == 0 or float(tail.mean()) < -45.0:
                                # include the wobble-decay past the peak
                                while (v + 1 < n
                                       and v - u < horizon + int(0.25 / W)
                                       and rms[v + 1] > -55.0):
                                    v += 1
                                spans.append((u, v))
                                u = v + 1
                                continue
        u += 1
    merged = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1] + 2:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _trim_bounds(rms, flat, win, sr, n):
    """Sample indices [a, b) keeping speech, cutting TTS edge garbage.

    Anchors are confident-speech windows (loud AND tone-like), with
    swelling-hum garbage spans masked out first (they are tonal enough to
    fake anchors — only their emerge-from-quiet + monotonic-rise shape gives
    them away). The end walks forward from the last anchor, keeping a natural
    tail (echo, breathy decay — these always FALL), but stopping at harsh
    bursts re-appearing from quiet, re-ascents after a real lull, the tail
    dying out, or a stationary hum/hiss. The quiet-guards keep real final
    fricatives safe: they are continuous with speech, with no lull before
    them. Only edges are ever trimmed — the interior is never touched.
    Returns None when no speech is found (caller keeps the input unchanged).
    """
    import numpy as np
    W = _WIN_S
    anchor = (rms > _ANCHOR_RMS_DB) & (flat < _ANCHOR_FLAT)
    for u, v in _swell_spans(rms):
        anchor[u:v + 1] = False
    idx = np.where(anchor)[0]
    if len(idx) == 0:
        return None
    first, last = int(idx[0]), int(idx[-1])

    end_w = last
    max_end = min(len(rms) - 1, last + int(1.00 / W))
    mature_at = last + int(0.25 / W)   # a natural tail always gets this much
    while end_w < max_end:
        j = end_w + 1
        # harsh burst re-appearing from quiet — never a natural tail
        if flat[j] > 0.65 and rms[j] > -45.0 and _emerged_from_quiet(j, rms):
            break
        # re-ascent after a real lull: swelling hums and late bursts.
        # (the lull-guard protects breathy dips, which never bottom out)
        if j > last + 2:
            floor = float(np.min(rms[last + 2:j + 1]))
            if floor < -50.0 and rms[j] > floor + 10.0 and rms[j] > -45.0:
                break
        if j > mature_at:
            if rms[j] < -48.0:
                break                  # tail died out
            w0 = max(0, j - int(0.30 / W))
            if float(np.var(rms[w0:j + 1])) < 1.5:
                break                  # stationary hum/hiss, not syllables
        end_w = j

    # start: tight onset pad, pushed past leading pre-speech hum.
    # Measured 27 Sep 2026 on a real Hindi unit: the model leaves a tonal
    # hum ramping -57 -> -44 -> -40 dB in the 60 ms before speech hits at
    # -23 dB. The old 0.25 s pad kept that hum. Now: 20 ms pad, but if the
    # pad window is low-level (< -35 dB) and speech jumps > 12 dB into the
    # first anchor, the pad is hum, not onset — cut it. Soft natural
    # onsets rise gradually (< 12 dB jump), so they keep their pad.
    start_w = max(0, first - int(0.02 / W))
    if start_w < first and rms[start_w] < -35.0 and rms[first] - rms[start_w] > 12.0:
        start_w = first
    for h in np.where((flat > 0.65) & (rms > -45.0))[0]:
        h = int(h)
        if h < first:
            start_w = max(start_w, h + int(0.06 / W))
        else:
            break

    a = start_w * win
    b = min(n, (end_w + 1) * win)
    if b - a < int(sr * 0.2):
        return None
    return a, b


def trim_unit(src: str, dst: str) -> str:
    """Cut a TTS unit's leading/trailing garbage hiss.

    Runs on every downloaded unit before it reaches the cache, so stitched
    beats get clean digital-silence pauses instead of seconds of model
    noise. Never destroys audio: any unexpected format, no speech found, or
    an implausible result copies the input through unchanged.
    """
    import numpy as np
    import wave

    try:
        with wave.open(src, "rb") as w:
            nch, sw, sr, n, _, _ = w.getparams()
            raw = w.readframes(n)
        if sw != 2 or sr < 8000 or n < sr // 5:
            raise ValueError("unexpected format")
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        mono = x.reshape(-1, nch).mean(axis=1) if nch > 1 else x
        rms, flat, win, _ = _frame_stats(mono, sr)
        bounds = _trim_bounds(rms, flat, win, sr, len(mono))
        if bounds is None:
            raise ValueError("no speech found")
        a, b = bounds
        frames = x[a * nch:b * nch].reshape(-1, nch).astype(np.float32)
        f = int(sr * 0.005)  # 5 ms anti-click fades at the cut points
        if len(frames) > 2 * f:
            ramp = np.linspace(0, 1, f)[:, None]
            frames[:f] *= ramp
            frames[-f:] *= ramp[::-1]
        with wave.open(dst, "wb") as w:
            w.setnchannels(nch)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((np.clip(frames, -1, 1) * 32767).astype(np.int16).tobytes())
        return dst
    except Exception:
        shutil.copy(src, dst)
        return dst


# ---------------------------------------------------------------- CLI

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print(__doc__)
        print("usage: python audioPolish.py <input.wav> <output.wav>")
        print("\n--- pause map ---")
        print(json.dumps(PAUSE_AFTER, indent=2))
        print("\n--- example instruct from voiceMetadata ---")
        print(build_instruct({
            "emotion": "curious", "energy": 0.75, "pace": "medium",
            "emphasis": ["राज़", "नब्बे परसेंट"],
            "pause_after_words": ["छुपा हुआ है"],
        }))
        print("\n--- example split ---")
        demo = ("दोस्तों, फिल्म की शुरुआत में ही एक ऐसा राज़ छुपा हुआ है जो नब्बे परसेंट "
                "लोगों ने नोटिस ही नहीं किया। कैमरा जब पहली बार उस कमरे में जाता है, तब "
                "दीवार पर एक तस्वीर दिखती है। और वही तस्वीर आगे पूरी कहानी को बदल देती है।")
        for u in apply_beat_hints(split_for_speech(demo), beat="REVEAL"):
            print(f"  [{u['pause']:.2f}s after]  {u['text'][:58]}")
        sys.exit(0)

    polish(sys.argv[1], sys.argv[2])
    print(f"polished -> {sys.argv[2]}")
