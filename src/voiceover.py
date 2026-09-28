"""
voiceover.py — production Hindi voiceover against the local Gradio TTS app.

Backend: "Chatterbox Turbo TTS" Gradio app at http://127.0.0.1:7861/,
endpoint /generate_multilingual_speech (the Multilingual TTS tab).
This replaced the old Voicebox REST backend, whose v0.5.0 broke server-side.

Transport detail that matters: we drive the Gradio queue over raw HTTP
(POST /gradio_api/queue/join, then read the SSE stream on
/gradio_api/queue/data) — exactly what the browser tab does. gradio_client
was tried first and rejected: it validates voice_name against the endpoint's
*static* Literal choices (the French defaults baked in at app load), so it
can never send Dad's Hindi voice "anokhi". Raw HTTP has no such check, and
the server accepts the voice the same way the UI does.

Dad's exact UI settings are the defaults (env-overridable):
    voice "anokhi", language hi, exaggeration 1, temperature 1,
    seed 1 (fixed => deterministic re-renders), cfgw 0.3

One wrinkle the UI hides: the voice dropdown's real choice values carry
suffixes ("anokhi" is really "anokhi ♂️"), and the server validates against
them. The voice name is therefore resolved against the app's live Hindi
voice list at runtime (read out of the language pre-select's update
payload), so Dad keeps writing the short name.

Speed, measured on his machine: 121 chars -> 114s -> 9s audio, i.e. RTF ~12.7.
A 25-minute narration is ~5.3 hours of compute. That single number drives the
design: everything is content-addressed, cached, and resumable, and nothing is
ever synthesised twice. Queue it overnight.

Requires: the Gradio app running, ffmpeg on PATH. No network beyond localhost.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, asdict

import requests

# Windows consoles default to cp1252, which crashes printing Devanagari
# narration text. Replace unprintable chars with ? instead of dying mid-run.
try:
    if sys.stdout is not None:
        sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

try:
    from audioPolish import (
        polish, split_for_speech, apply_beat_hints, build_instruct, stitch,
        PAUSE_AFTER, trim_unit,
    )
except ImportError:  # allow `python -m src.voiceover`
    from .audioPolish import (  # type: ignore
        polish, split_for_speech, apply_beat_hints, build_instruct, stitch,
        PAUSE_AFTER, trim_unit,
    )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "voice_cache")
OUT_DIR = os.path.join(BASE_DIR, "voice_out")
DATA_DIR = os.path.join(BASE_DIR, "data")
MANIFEST = os.path.join(DATA_DIR, "voiceover_manifest.json")

# ---------------------------------------------------------------- backend config
# Dad's exact Gradio UI settings. Override in .env if he retunes the UI.

GRADIO_URL = os.getenv("GRADIO_URL", "http://127.0.0.1:7861/")
GRADIO_VOICE = os.getenv("GRADIO_VOICE_NAME", "anokhi")
GRADIO_EXAGGERATION = float(os.getenv("GRADIO_EXAGGERATION", "1"))
GRADIO_TEMPERATURE = float(os.getenv("GRADIO_TEMPERATURE", "1"))
GRADIO_SEED = int(float(os.getenv("GRADIO_SEED", "1")))
GRADIO_CFGW = float(os.getenv("GRADIO_CFGW", "0.3"))
GRADIO_LANG = "hi"

APP = GRADIO_URL.rstrip("/")
API = APP + "/gradio_api"

BACKEND_LABEL = "gradio-multilingual"

# Measured: 121 chars -> 114.2s wall -> 9.0s audio => RTF 12.7.
# Used only to print an honest ETA before a long run. --bench refines it.
MEASURED_RTF = float(os.getenv("GRADIO_RTF", "14.8"))

# Hindi narration rate: 121 chars -> 9.0s, i.e. ~13.4 chars/sec. Working figure.
CHARS_PER_SECOND = 13.4

UNIT_TIMEOUT = 1800  # one sentence-unit genuinely can take minutes on CPU


class VoiceoverError(RuntimeError):
    pass


# ---------------------------------------------------------------- cache

def _cache_key(text: str, *, seed: int, voice: str) -> str:
    """Content address for one synthesised line.

    At ~13x realtime, a cache hit is worth minutes. The key covers everything
    that changes the audio, so retuning a setting re-renders, but editing one
    sentence in a script re-renders one sentence, not the whole narration.
    """
    blob = json.dumps({
        "text": text.strip(), "backend": BACKEND_LABEL,
        "voice": voice, "language": GRADIO_LANG,
        "exaggeration": GRADIO_EXAGGERATION, "temperature": GRADIO_TEMPERATURE,
        "seed": seed, "cfgw": GRADIO_CFGW,
        "v": 4,   # bump to invalidate everything after a chain change
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


def _cached_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.wav")


# ---------------------------------------------------------------- gradio queue transport (raw HTTP)

_API_INDEXES = None  # (fn_index_generate, fn_index_lang)


def _api_indexes() -> tuple[int, int | None]:
    """Resolve the queue function indexes from the app's /config.

    Connecting here is also the health check: if this fails, the app isn't up.
    Endpoint names are matched loosely (leading slash or not) because /config
    doesn't always spell them the way the docs page does.
    """
    global _API_INDEXES
    if _API_INDEXES is not None:
        return _API_INDEXES
    try:
        cfg = requests.get(f"{APP}/config", timeout=15).json()
    except Exception as e:
        raise VoiceoverError(
            f"can't reach the Gradio TTS app at {APP} — "
            f"is it running? ({e})")
    names: dict[str, int] = {}
    for dep in cfg.get("dependencies", []):
        raw = str(dep.get("api_name") or "")
        norm = raw.strip().lstrip("/").lower()
        if norm and norm not in names:
            names[norm] = dep.get("id")
    gen_idx = names.get("generate_multilingual_speech")
    # The language -> voice-list updater: not "the first /lambda", but the
    # lambda actually wired to the language dropdown's change event (found via
    # component labels in /config). Calling a random lambda is worse than
    # calling none — a wrong one can error or poke unrelated UI state.
    lang_idx = _find_lang_lambda(cfg)
    if gen_idx is None:
        seen = sorted(names)[:40]
        raise VoiceoverError(
            "the Gradio app is up but I can't find its multilingual TTS "
            f"endpoint. Endpoints I see: {seen} — paste this list back "
            "and I'll point at the right one.")
    _API_INDEXES = (gen_idx, lang_idx)
    return _API_INDEXES


def _find_lang_lambda(cfg: dict) -> int | None:
    """fn_index of the lambda wired to the language dropdown's change event.

    /config lists every component (with its label) and every dependency (with
    its input component ids). The language dropdown's change handler is the
    dependency whose inputs include that dropdown. Returns None when it can't
    be identified — the caller then skips the pre-select instead of firing a
    random lambda.
    """
    lang_ids = set()
    for comp in cfg.get("components", []):
        props = comp.get("props") or {}
        label = str(props.get("label") or "")
        if "language" in label.lower():
            lang_ids.add(comp.get("id"))
    if not lang_ids:
        return None
    for dep in cfg.get("dependencies", []):
        raw = str(dep.get("api_name") or "").strip().lstrip("/").lower()
        if not raw.startswith("lambda"):
            continue
        inputs = dep.get("inputs") or []
        if any(i in lang_ids for i in inputs):
            return dep.get("id")
    return None


_RESOLVED_VOICE: str | None = None


def _extract_choices(obj):
    """Find a choice list inside the language lambda's update payload.

    The Hindi voice list only exists after the language pre-select, and the
    lambda answers with a Dropdown update carrying the live choices. Shapes
    vary (plain strings or [label, value] pairs, nested in the update dict),
    so this digs liberally.
    """
    if isinstance(obj, dict):
        ch = obj.get("choices")
        if isinstance(ch, list) and ch:
            return ch
        for v in obj.values():
            r = _extract_choices(v)
            if r:
                return r
    elif isinstance(obj, (list, tuple)):
        if obj and all(isinstance(x, str) for x in obj):
            return list(obj)
        if obj and all(isinstance(x, (list, tuple)) and len(x) == 2
                       and all(isinstance(y, str) for y in x) for x in obj):
            return list(obj)
        for v in obj:
            r = _extract_choices(v)
            if r:
                return r
    return []


def _match_choice(choices, want: str) -> str:
    """Dad's short name -> the dropdown's real choice value.

    The UI shows 'anokhi' but the choice value is 'anokhi ♂️'. Match exact
    first, then emoji/whitespace-insensitive, then prefix. No match returns
    `want` unchanged so the server complains loudly with its real list.
    """
    vals = [str(c[1]) if isinstance(c, (list, tuple)) and len(c) == 2 else str(c)
            for c in choices]
    if want in vals:
        return want
    norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())
    for v in vals:
        if norm(v) == norm(want):
            return v
    for v in vals:
        if norm(v).startswith(norm(want)):
            return v
    return want


def _voice_from_error(msg: str, want: str) -> str | None:
    """Last resort: parse the real choices out of the server's validation
    error ('Value: X is not in the list of choices: [...]'). Returns the
    matched choice, or None when there's nothing to fix."""
    m = re.search(r"not in the list of choices:\s*(\[.*\])", msg, re.S)
    if not m:
        return None
    try:
        choices = ast.literal_eval(m.group(1))
    except Exception:
        return None
    if not isinstance(choices, list):
        return None
    fixed = _match_choice(choices, want)
    return fixed if fixed != want else None


def _resolved_voice() -> str:
    """Dad's GRADIO_VOICE_NAME -> the live choice value the server accepts.

    Name-learner only, used for cache-key computation before any generate
    call: fire the language pre-select in a throwaway session, read the Hindi
    voice list out of its update payload, and match. (The pre-select that
    matters for validation happens per generate call inside synth(), in the
    same session — that state does not stick across sessions.) Falls back to
    the short name; the server then rejects it loudly, listing its real
    choices, and synth() parses that error and retries once.
    """
    global _RESOLVED_VOICE
    if _RESOLVED_VOICE is None:
        _RESOLVED_VOICE = GRADIO_VOICE
        try:
            _, lang_idx = _api_indexes()
            if lang_idx is not None:
                session = uuid.uuid4().hex
                ev = _queue_join(lang_idx, [GRADIO_LANG], session)
                out = _await_event(session, ev, 120)
                _RESOLVED_VOICE = _match_choice(
                    _extract_choices(out or []), GRADIO_VOICE)
        except Exception:
            pass
    return _RESOLVED_VOICE


def _queue_join(fn_index: int, data: list, session: str) -> str:
    r = requests.post(f"{API}/queue/join", json={
        "data": data,
        "event_data": None,
        "fn_index": fn_index,
        "session_hash": session,
    }, timeout=30)
    r.raise_for_status()
    return r.json()["event_id"]


_PENDING = object()


def _await_event(session: str, event_id: str, timeout: float):
    """Read the SSE stream until our job completes or errors. Returns the
    endpoint's output data list.

    Wire protocol (verified against a live Gradio server): the stream carries
    no `event:` lines — each `data:` line is JSON with a {"msg": ...} field:
    estimation -> process_starts -> process_completed -> close_stream.
    Completion is {"msg":"process_completed","event_id":...,
    "output":{"data":[...]}}.
    """
    url = f"{API}/queue/data"
    deadline = time.time() + timeout
    recent: list[tuple[str, str]] = []  # last messages, for diagnostics
    with requests.get(url, params={"session_hash": session},
                      headers={"Accept": "text/event-stream"},
                      stream=True, timeout=60) as r:
        r.raise_for_status()
        sse_event: str | None = None
        buf: list[str] = []
        for line in r.iter_lines(decode_unicode=True):
            if time.time() > deadline:
                raise VoiceoverError(
                    "TTS timed out waiting for the Gradio app "
                    f"({timeout:.0f}s) — is it stuck?")
            if line is None:
                continue
            s = line.strip()
            if not s:
                if buf:
                    raw = "\n".join(buf)
                    buf = []
                    out = _handle_message(raw, event_id, sse_event, recent)
                    if out is not _PENDING:
                        return out
                sse_event = None
                continue
            if s.startswith(":"):        # SSE comment / heartbeat
                continue
            if s.startswith("event:"):
                sse_event = s[6:].strip()
            elif s.startswith("data:"):
                buf.append(s[5:].lstrip())
    raise VoiceoverError(
        "TTS stream ended without a result "
        f"(event_id={event_id}). Last messages: {recent[-8:]}")


def _handle_message(raw: str, event_id: str, sse_event: str | None,
                    recent: list) -> object:
    """One blank-line-delimited SSE block -> output payload or _PENDING."""
    try:
        m = json.loads(raw)
    except Exception:
        return _PENDING
    if isinstance(m, dict) and "msg" in m:
        msg, eid = m.get("msg"), m.get("event_id")
        recent.append((str(msg), raw[:200]))
        del recent[:-8]
        if eid != event_id:
            return _PENDING
        if msg == "process_completed":
            if not m.get("success", True):
                # A failing job comes back as success:false with NO data key
                # (verified against a live server) — surface it, don't
                # collapse it into a mysterious None.
                raise VoiceoverError(
                    "TTS job failed server-side "
                    f"(event_id={event_id}). Raw: {raw[:300]}")
            out = m.get("output")
            data = out.get("data") if isinstance(out, dict) else out
            if not data:
                raise VoiceoverError(
                    "TTS completed but returned no audio data "
                    f"(event_id={event_id}). Raw: {raw[:300]}")
            return data
        if msg == "error":
            raise VoiceoverError(
                f"TTS job failed: {str(m.get('error', m))[:300]}")
        return _PENDING
    # Fallback: very old Gradio used `event: complete` lines instead of msg.
    if sse_event in ("complete", "error"):
        eid, output = _unwrap(m)
        if eid != event_id:
            return _PENDING
        if sse_event == "error":
            raise VoiceoverError(f"TTS job failed: {str(output)[:300]}")
        return output
    return _PENDING


def _unwrap(payload):
    """(event_id, output) from the various shapes Gradio has used."""
    if isinstance(payload, dict):
        return payload.get("event_id"), payload.get("output",
                                                     payload.get("data"))
    if isinstance(payload, list):
        if len(payload) == 2 and isinstance(payload[0], str):
            return payload[0], payload[1]
        if len(payload) == 1:
            return _unwrap(payload[0])
    return None, None


def _download(url_path: str, dst: str) -> None:
    # The app sometimes returns an absolute URL, sometimes a path — handle
    # both instead of blindly prepending the host (which produced
    # "127.0.0.1:7861http://..." and a parse error).
    url = (url_path if url_path.startswith(("http://", "https://"))
           else APP + url_path)
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    tmp = dst + ".part"
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(65536):
            f.write(chunk)
    os.replace(tmp, dst)     # atomic, so a crash never leaves a half file cached


def health() -> dict:
    """Verify the Gradio app is up and the endpoint exists."""
    gen_idx, _ = _api_indexes()
    return {"backend": BACKEND_LABEL, "url": APP,
            "voice": GRADIO_VOICE, "fn_index": gen_idx}


def synth(text: str, *, seed: int = GRADIO_SEED, quiet: bool = True) -> str:
    """Synthesise one speakable unit via the Gradio app. Returns a wav path.

    Cached by content hash, so a second run only renders what changed.
    One retry on transport failure; a real TTS failure raises.
    """
    global _RESOLVED_VOICE
    text = (text or "").strip()
    if not text:
        raise VoiceoverError("nothing to say")

    voice = _resolved_voice()
    key = _cache_key(text, seed=seed, voice=voice)
    dst = _cached_path(key)
    if os.path.exists(dst) and os.path.getsize(dst) > 1024:
        return dst

    os.makedirs(CACHE_DIR, exist_ok=True)
    last: Exception | None = None
    output = None
    for _attempt in range(2):
        try:
            gen_idx, lang_idx = _api_indexes()
            session = uuid.uuid4().hex
            # Same session, language first — exactly like the browser tab.
            # This pre-select is what loads the Hindi voice list the server
            # validates against, and that state is per-session: verified
            # 27 Sep 2026 that a pre-select in a *different* session does NOT
            # stick (the generate then sees the French defaults and rejects
            # the voice). The live choices also resolve the real voice value
            # ("anokhi" -> "anokhi ♂️").
            voice = _RESOLVED_VOICE or GRADIO_VOICE
            if lang_idx is not None:
                lang_event = _queue_join(lang_idx, [GRADIO_LANG], session)
                lang_out = _await_event(session, lang_event, 120)
                voice = _match_choice(_extract_choices(lang_out or []), voice)
                _RESOLVED_VOICE = voice
            event_id = _queue_join(gen_idx, [
                text, voice, GRADIO_LANG,
                GRADIO_EXAGGERATION, GRADIO_TEMPERATURE,
                seed, GRADIO_CFGW,
            ], session)
            output = _await_event(session, event_id, UNIT_TIMEOUT)
            break
        except VoiceoverError as ve:
            # Safety net: if the server rejects the voice name, its error
            # lists the real choices — resolve from it and retry once.
            fixed = (_voice_from_error(str(ve), GRADIO_VOICE)
                     if "not in the list of choices" in str(ve) else None)
            if fixed and fixed != voice:
                voice = _RESOLVED_VOICE = fixed
                key = _cache_key(text, seed=seed, voice=voice)
                dst = _cached_path(key)
                if os.path.exists(dst) and os.path.getsize(dst) > 1024:
                    return dst
                if not quiet:
                    print(f"      voice resolved to {voice!r}, retrying...")
                continue
            raise
        except Exception as e:
            last = e
            if not quiet:
                print(f"      transport hiccup ({e}), retrying...")
            time.sleep(5)
    if output is None:
        raise VoiceoverError(f"TTS request failed twice ({last})")

    try:
        _progress, fileinfo, status = output
    except (TypeError, ValueError):
        raise VoiceoverError(
            f"unexpected TTS response shape: {str(output)[:200]}")

    status_s = str(status or "")
    if any(w in status_s.lower()
           for w in ("fail", "error", "cancel", "exception")):
        raise VoiceoverError(f"TTS job reported failure: {status_s[:300]}")

    url_path = (fileinfo or {}).get("url") if isinstance(fileinfo, dict) else None
    if not url_path:
        raise VoiceoverError(
            f"TTS returned no audio file (status={status_s[:200]!r})")

    _download(url_path, tmp_raw := dst + ".raw.wav")
    try:
        # The TTS renders seconds of garbage hiss/bursts around the words —
        # trim each unit before it reaches the cache, so stitched beats get
        # clean digital-silence pauses instead of model noise.
        trim_unit(tmp_raw, dst)
    finally:
        if os.path.exists(tmp_raw):
            os.remove(tmp_raw)
    if os.path.getsize(dst) < 1024:
        os.remove(dst)
        raise VoiceoverError("TTS returned an empty audio file")
    return dst


# ---------------------------------------------------------------- one scene

@dataclass
class SceneAudio:
    beat: str
    file: str
    duration: float
    units: int
    engine: str
    instruct: str = ""
    cached_units: int = 0

    def dict(self) -> dict:
        return asdict(self)


def wav_duration(path: str) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, check=True)
        return round(float(out.stdout.strip()), 3)
    except Exception:
        return 0.0


def synth_scene(scene: dict, *, index: int = 0,
                seed: int = GRADIO_SEED) -> SceneAudio:
    """Render one script beat: split, voice each line, stitch, polish.

    `scene` is a beat straight out of the script engine — it already carries
    voiceoverText and voiceMetadata. The Gradio multilingual endpoint takes no
    instruct parameter, so performance direction is shaped through sentence
    structure and authored pauses (the metadata is still logged, and kept in
    the script for the future).
    """
    beat = str(scene.get("beat") or scene.get("sceneType") or f"scene{index}")
    text = (scene.get("voiceoverText") or scene.get("narration") or "").strip()
    if not text:
        raise VoiceoverError(f"beat {beat} has no voiceoverText")

    meta = scene.get("voiceMetadata") or {}
    instruct = build_instruct(meta) if meta else ""

    units = apply_beat_hints(split_for_speech(text), beat=beat)
    print(f"\n  [{index}] {beat} — {len(units)} units, "
          f"{len(text)} chars")
    if instruct:
        print(f"      direction (not sent to TTS): {instruct[:96]}")

    os.makedirs(OUT_DIR, exist_ok=True)
    segments: list[dict] = []
    cached = 0
    voice = _resolved_voice()  # same key synth() will use, for the check below

    for i, unit in enumerate(units):
        key = _cache_key(unit["text"], seed=seed, voice=voice)
        was_cached = os.path.exists(_cached_path(key))

        t0 = time.time()
        path = synth(unit["text"], seed=seed)

        if was_cached:
            cached += 1
            print(f"      {i + 1}/{len(units)} cached   {unit['text'][:44]}")
        else:
            print(f"      {i + 1}/{len(units)} {time.time() - t0:5.1f}s  "
                  f"{unit['text'][:44]}")

        segments.append({"file": path, "pause": unit["pause"]})

    raw = os.path.join(OUT_DIR, f"{index:02d}_{beat}_raw.wav")
    final = os.path.join(OUT_DIR, f"{index:02d}_{beat}.wav")
    stitch(segments, raw)
    polish(raw, final)
    os.remove(raw)

    dur = wav_duration(final)
    print(f"      -> {os.path.basename(final)}  {dur:.2f}s "
          f"({cached}/{len(units)} from cache)")

    return SceneAudio(beat=beat, file=final, duration=dur, units=len(units),
                      engine=BACKEND_LABEL, instruct=instruct,
                      cached_units=cached)


# ---------------------------------------------------------------- whole script

def estimate(scenes: list) -> tuple[float, float]:
    """(audio_seconds, compute_minutes) — printed before committing hours."""
    chars = sum(len((s.get("voiceoverText") or s.get("narration") or ""))
                for s in scenes)
    audio_s = chars / CHARS_PER_SECOND
    compute_s = audio_s * MEASURED_RTF
    return audio_s, compute_s / 60.0


def synth_script(script: dict, *, seed: int = GRADIO_SEED) -> dict:
    """Voice a whole script. Writes a timing manifest for the assembly step.

    The manifest is the contract with the ffmpeg stage: it needs exact per-beat
    narration durations to lay footage against, and guessing them is how
    voiceover drifts out of sync with picture.
    """
    scenes = script.get("scenes") or script.get("beats") or []
    if not scenes:
        raise VoiceoverError("script has no scenes")

    audio_s, compute_min = estimate(scenes)
    print("=" * 68)
    print(f"  VOICEOVER — {len(scenes)} beats  ({BACKEND_LABEL})")
    print(f"  voice     : {GRADIO_VOICE}  hi  "
          f"exagg={GRADIO_EXAGGERATION} temp={GRADIO_TEMPERATURE} "
          f"seed={GRADIO_SEED} cfgw={GRADIO_CFGW}")
    print(f"  estimated audio   : {audio_s / 60:.1f} min")
    print(f"  estimated compute : {compute_min:.0f} min "
          f"(RTF {MEASURED_RTF} on CPU, minus cache hits)")
    print("=" * 68)

    h = health()
    print(f"  backend={h.get('backend')} url={h.get('url')}")

    started = time.time()
    rendered: list[SceneAudio] = []
    cursor = 0.0
    timeline: list[dict] = []

    for i, scene in enumerate(scenes):
        sa = synth_scene(scene, index=i, seed=seed)
        rendered.append(sa)
        timeline.append({
            "beat": sa.beat,
            "file": sa.file,
            "start": round(cursor, 3),
            "end": round(cursor + sa.duration, 3),
            "duration": sa.duration,
        })
        cursor += sa.duration
        _write_manifest(script, rendered, timeline, cursor, started)

    full = os.path.join(OUT_DIR, "narration_full.wav")
    stitch([{"file": s.file, "pause": PAUSE_AFTER["paragraph"]}
            for s in rendered], full)

    manifest = _write_manifest(script, rendered, timeline, cursor, started,
                              full=full)

    wall = (time.time() - started) / 60
    print("\n" + "=" * 68)
    print(f"  narration : {full}")
    print(f"  duration  : {cursor / 60:.2f} min")
    print(f"  wall time : {wall:.1f} min "
          f"(actual RTF {(time.time() - started) / max(cursor, 1):.1f})")
    print(f"  manifest  : {MANIFEST}")
    print("=" * 68)
    return manifest


def _write_manifest(script: dict, rendered: list, timeline: list,
                    total: float, started: float, full: str = "") -> dict:
    os.makedirs(DATA_DIR, exist_ok=True)
    manifest = {
        "title": script.get("title") or script.get("topic") or "",
        "language": "hi",
        "voice": GRADIO_VOICE,
        "backend": BACKEND_LABEL,
        "settings": {
            "exaggeration": GRADIO_EXAGGERATION,
            "temperature": GRADIO_TEMPERATURE,
            "seed": GRADIO_SEED,
            "cfgw": GRADIO_CFGW,
        },
        "narration_file": full,
        "total_duration": round(total, 3),
        "beats": [s.dict() for s in rendered],
        "timeline": timeline,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_minutes": round((time.time() - started) / 60, 2),
        "complete": bool(full),
    }
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


# ---------------------------------------------------------------- bench

BENCH_TEXT = ("दोस्तों, फिल्म की शुरुआत में ही एक ऐसा राज़ छुपा हुआ है "
              "जो नब्बे परसेंट लोगों ने नोटिस ही नहीं किया।")


def bench() -> None:
    """One honest measurement: voice the 100-char Hindi probe with Dad's
    exact settings and report the real RTF, so the ETA before a long run is
    a measurement, not a guess."""
    print("=" * 68)
    print("  VOICEOVER BENCH")
    print("=" * 68)
    h = health()
    print(f"  backend={h.get('backend')} voice={h.get('voice')}")
    print(f"  settings: exagg={GRADIO_EXAGGERATION} temp={GRADIO_TEMPERATURE} "
          f"seed={GRADIO_SEED} cfgw={GRADIO_CFGW}")
    print(f"  text: {len(BENCH_TEXT)} chars\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    results: list[dict] = []
    t0 = time.time()
    try:
        raw = synth(BENCH_TEXT, quiet=False)
        wall = time.time() - t0
        dst = os.path.join(OUT_DIR, "bench_gradio.wav")
        shutil.copyfile(raw, dst)
        polished = os.path.join(OUT_DIR, "bench_gradio_polished.wav")
        polish(dst, polished)

        dur = wav_duration(polished) or wav_duration(dst)
        rtf = wall / dur if dur else 0
        print(f"      OK  {dur:.1f}s audio in {wall:.0f}s  RTF {rtf:.1f}")
        if rtf:
            print(f"          12-min narration would take "
                  f"{rtf * 720 / 60:.0f} min")
        print(f"          {polished}")
        results.append({"backend": BACKEND_LABEL, "voice": GRADIO_VOICE,
                        "ok": True, "rtf": round(rtf, 2),
                        "wall": round(wall, 1), "duration": dur,
                        "file": polished})
    except VoiceoverError as e:
        print(f"      FAIL  {e}")
        results.append({"backend": BACKEND_LABEL, "voice": GRADIO_VOICE,
                        "ok": False, "error": str(e)[:300]})

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "voiceover_bench.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("=" * 68)
    print("  SUMMARY")
    for r in results:
        if r.get("ok"):
            print(f"    gradio-multilingual  RTF {r['rtf']:<7} "
                  f"12min -> {r['rtf'] * 12:.0f} min")
        else:
            print("    gradio-multilingual  FAILED")
    print("\n  Listen to voice_out/bench_gradio_polished.wav — if the voice")
    print("  is right, the pipeline is ready. Speed only breaks the tie.")
    print("=" * 68)


# ---------------------------------------------------------------- CLI

def _demo_script() -> dict:
    return {"title": "demo", "scenes": [
        {"sceneType": "COLD_OPEN",
         "voiceoverText": ("दोस्तों, इस फिल्म का पहला सीन ही झूठ बोलता है। "
                           "और तुमने उसे बिल्कुल नोटिस नहीं किया।"),
         "voiceMetadata": {"emotion": "curious", "energy": 0.85,
                           "pace": "fast", "emphasis": ["झूठ"]}},
        {"sceneType": "REVEAL",
         "voiceoverText": ("अब ध्यान से देखो। दीवार पर लगी वो तस्वीर, "
                           "जो शुरू में बेमतलब लगती थी। वही पूरी कहानी "
                           "का असली राज़ है।"),
         "voiceMetadata": {"emotion": "conspiratorial", "energy": 0.6,
                           "pace": "slow", "emphasis": ["असली राज़"],
                           "pause_after_words": ["बेमतलब लगती थी"]}},
    ]}


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        print("usage:")
        print("  python src/voiceover.py --bench")
        print("       measure the real RTF with your exact Gradio settings")
        print("  python src/voiceover.py --say \"हिंदी टेक्स्ट\"")
        print("       voice one line, polished, for a quick listen")
        print("  python src/voiceover.py --demo")
        print("       render a 2-beat demo script end to end")
        print("  python src/voiceover.py --script path/to/script.json")
        print("       voice a real script and write the timing manifest")
        print("  env: GRADIO_URL GRADIO_VOICE_NAME GRADIO_EXAGGERATION")
        print("       GRADIO_TEMPERATURE GRADIO_SEED GRADIO_CFGW")
        sys.exit(0)

    if args[0] == "--bench":
        bench()

    elif args[0] == "--say":
        if len(args) < 2:
            print("give me something to say")
            sys.exit(1)
        raw = synth(args[1], quiet=False)
        os.makedirs(OUT_DIR, exist_ok=True)
        out = os.path.join(OUT_DIR, "say.wav")
        polish(raw, out)
        print(f"\n  voice={GRADIO_VOICE}  {out}  {wav_duration(out):.2f}s")

    elif args[0] == "--demo":
        synth_script(_demo_script())

    elif args[0] == "--script":
        with open(args[1], encoding="utf-8") as f:
            synth_script(json.load(f))

    else:
        print(f"unknown option {args[0]!r} — try --help")
        sys.exit(1)
