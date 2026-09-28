"""
script_writer.py — turn a sourceRead beat-sheet into a Hindi narration script
where every beat is tagged with the exact film span it describes.

Pipeline position:
    sourceRead -> script_writer -> voiceover -> aligner -> assemble

This is the module the whole "not AI slop" promise hangs on. The slop version
writes narration from a title and hopes the footage matches. This version does
three passes over Pollinations (OpenAI-compatible chat completions):

  A. PLAN   — the brief's windows become a beat plan: each beat carries
              source_span (absolute film seconds it describes) and visual_style
              (how aligner should shoot it). Chronological, non-overlapping.
  B. WRITE  — each beat's summary + the real dialogue inside its span become
              Hindi voiceoverText (Devanagari) with voiceMetadata performance
              direction for voiceover.py.
  C. JUDGE  — one coherence/engagement pass; beats scoring below the bar get
              exactly one rewrite. Bounded: quality without an infinite loop.

Output script.json — the contract voiceover.py and aligner.py both read:
  {"title_working": ..., "language": "hi", "film": ..., "runtime_s": ...,
   "beats_planned": N, "judge": {...},
   "scenes": [{"beat": "BEAT_01", "sceneType": "COLD_OPEN",
               "source_span": [0.0, 95.0], "visual_style": "mixed",
               "summary": "...",
               "voiceoverText": "दोस्तों ...",
               "voiceMetadata": {"emotion": "curious", "energy": 0.85,
                                 "pace": "fast", "emphasis": ["झूठ"],
                                 "pause_after_words": ["..."]}}]}

Design rules:
  * Stdlib + requests only. No numpy, no torch — this runs on the same box as
    everything else and must not disturb the pinned TTS environment.
  * Narration is Devanagari Hindi. Prompts to the model are English (LLMs are
    stronger at following structural instructions in English); only the
    voiceoverText it returns is Hindi.
  * Every beat's source_span is validated: inside the film, start < end,
    chronological and non-overlapping. A beat that fails validation fails the
    run loudly — a silently mistagged beat is how slop is born.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

# Windows consoles default to cp1252, which crashes printing non-ASCII
# (Devanagari narration, transcript dialogue). Replace unprintable chars
# with ? instead of dying mid-run.
try:
    if sys.stdout is not None:
        sys.stdout.reconfigure(errors="replace")
except Exception:
    pass


try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

# ---------------------------------------------------------------- tunables

POLLINATIONS_URL = os.getenv("POLLINATIONS_URL",
                             "https://gen.pollinations.ai/v1/chat/completions")
POLLINATIONS_MODEL = os.getenv("POLLINATIONS_MODEL", "openai")

# Hindi TTS lands around 800-900 chars/min; plan beats against this so the
# finished narration hits the target runtime instead of hoping it does.
CHARS_PER_MIN = int(os.getenv("SCRIPT_CHARS_PER_MIN", "850"))

SCENE_TYPES = ["COLD_OPEN", "SETUP", "RISING", "DIALOGUE", "ACTION",
               "REVEAL", "TWIST", "CLIMAX", "RESOLUTION"]
VISUAL_STYLES = ["action", "mixed", "dialogue", "reveal"]

JUDGE_BAR = 7          # below this on any axis -> one rewrite pass
MAX_REFINES = 1
WRITE_BATCH = 5        # beats per narration call


class ScriptError(RuntimeError):
    pass


# ---------------------------------------------------------------- LLM client

def _api_key() -> str:
    key = os.getenv("POLLINATIONS_API_KEY", "").strip()
    if not key:
        raise ScriptError(
            "POLLINATIONS_API_KEY is not set — put it in .env "
            "(see .env.example)")
    return key


def chat(messages: list, *, temperature: float = 0.7,
         max_tokens: int = 6000) -> str:
    """One chat-completions call. Retries once on transport/5xx errors."""
    if requests is None:
        raise ScriptError("the 'requests' package is required "
                          "(pip install requests)")
    headers = {"Authorization": f"Bearer {_api_key()}",
               "Content-Type": "application/json"}
    payload = {"model": POLLINATIONS_MODEL, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    last = None
    for attempt in range(2):
        try:
            r = requests.post(POLLINATIONS_URL, headers=headers,
                              json=payload, timeout=180)
            if r.status_code >= 500:
                last = f"HTTP {r.status_code}"
                time.sleep(5)
                continue
            if r.status_code != 200:
                raise ScriptError(
                    f"Pollinations HTTP {r.status_code}: {r.text[:300]}")
            return r.json()["choices"][0]["message"]["content"]
        except ScriptError:
            raise
        except Exception as e:  # transport error — one retry
            last = str(e)
            time.sleep(5)
    raise ScriptError(f"Pollinations call failed twice ({last})")


_NO_JSON = object()


def _try_parse(chunk: str):
    """json.loads + light repairs for the usual model sloppiness."""
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        pass
    # trailing commas: {"a": 1,} / [1, 2,]
    fixed = re.sub(r",\s*([}\]])", r"\1", chunk)
    # python literals some models emit
    fixed = re.sub(r"\bNone\b", "null", fixed)
    fixed = re.sub(r"\bTrue\b", "true", fixed)
    fixed = re.sub(r"\bFalse\b", "false", fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return _NO_JSON


def extract_json(text: str):
    """Pull the first JSON value out of model output (tolerates preamble).

    Uses proper bracket matching so a syntax error inside a big array doesn't
    degrade into returning some random inner object.
    """
    t = re.sub(r"```(?:json)?", "", text or "")
    starts = [i for i, c in enumerate(t) if c in "{["]
    if not starts:
        raise ScriptError("model returned no JSON at all")
    pairs = {"{": "}", "[": "]"}
    for s in starts:
        opener = t[s]
        closer = pairs[opener]
        depth = 0
        in_str = False
        esc = False
        for i in range(s, len(t)):
            c = t[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    val = _try_parse(t[s:i + 1])
                    if val is not _NO_JSON:
                        return val
                    break  # this opener's value is broken; try next opener
    raise ScriptError("model output was not parseable JSON")


def chat_json(messages: list, *, temperature: float = 0.7,
              max_tokens: int = 6000):
    """chat() + parse, with one repair retry that demands JSON only."""
    try:
        return extract_json(chat(messages, temperature=temperature,
                                  max_tokens=max_tokens))
    except ScriptError:
        fixed = messages + [{
            "role": "user",
            "content": ("Your last reply was not valid JSON. Reply again with "
                        "ONLY the JSON value, no commentary, no code fences.")}]
        return extract_json(chat(fixed, temperature=0.2,
                                  max_tokens=max_tokens))


# ---------------------------------------------------------------- pass A: plan

def _window_brief(w: dict) -> str:
    dlg = (w.get("dialogue") or "")[:420]
    return (f"[{w.get('i')}] {w.get('timecode')} energy={w.get('energy')} "
            f"speech={w.get('speech_seconds')}s shots={w.get('shot_count')} "
            f"dialogue: {dlg}")


PLAN_SYSTEM = """You are a story editor for Hindi "movie explained" YouTube videos.
You get a beat-sheet of a film: timecoded windows with energy, speech amount
and transcribed dialogue. You output a BEAT PLAN as JSON.

Rules:
- Cover the film's full arc: hook, setup, rising action, twists, climax, resolution.
- Each beat: {"beat": "BEAT_01", "sceneType": one of "COLD_OPEN", "SETUP", "RISING", "DIALOGUE", "ACTION", "REVEAL", "TWIST", "CLIMAX", "RESOLUTION",
  "source_span": [start_seconds, end_seconds], "visual_style": one of "action", "mixed", "dialogue", "reveal",
  "summary": "1-2 lines, English, what happens in this beat"}
- source_span uses ABSOLUTE film seconds. Spans must be chronological and non-overlapping.
  A span should be 60-240 seconds of film per beat.
- sceneType COLD_OPEN exactly once (first beat), RESOLUTION exactly once (last beat).
- COUNT CHECK: the film is RUNTIME seconds and you must produce exactly N beats,
  so each span averages ~RUNTIME/N seconds. Count your beats before replying —
  returning fewer than N is failure, not a shortcut.
- Every major plot twist gets its own beat.
- summary must be CONCRETE: who did what to whom and why, with names.
  "A tactical hostage release unfolds" is failure;
  "Captain Gao holds the sniper at gunpoint to force the convoy through" is correct.
  Each summary must connect causally to the previous beat's outcome.
- visual_style: action for fights/chases, dialogue for talky/emotional beats,
  reveal for twists and payoff moments, mixed otherwise.
- Reply with ONLY a JSON array of beats, no commentary. The JSON must be valid:
  no trailing commas, double quotes only."""


def plan_beats(brief: dict, *, target_beats: int) -> list:
    windows = brief.get("windows") or []
    if not windows:
        raise ScriptError("brief has no windows")
    runtime = float(brief.get("source_duration") or 0)
    body = "\n".join(_window_brief(w) for w in windows)
    user = (f"Film runtime: {runtime:.0f}s. Language: {brief.get('language')}.\n"
            f"Produce exactly {target_bets(target_beats)} beats "
            f"(~{runtime / max(1, target_beats):.0f}s of film per beat on average).\n\n"
            f"WINDOWS:\n{body}")
    beats = chat_json(
        [{"role": "system", "content": PLAN_SYSTEM},
         {"role": "user", "content": user}],
        temperature=0.6, max_tokens=8000)
    # The "count your beats" instruction sometimes makes the model wrap the
    # array in an object ({"beats": [...], "count": 35}). Unwrap it.
    if isinstance(beats, dict):
        for key in ("beats", "plan", "data"):
            if isinstance(beats.get(key), list):
                print(f"  (unwrapped planner response key {key!r})")
                beats = beats[key]
                break
    if not isinstance(beats, list) or not beats:
        raise ScriptError(
            "planner did not return a beat list; got: "
            f"{str(beats)[:300]}")
    return _validate_beats(beats, runtime, target_beats)


def target_bets(n: int) -> int:  # tiny helper, keeps the prompt line readable
    return n


def _coerce_choice(raw, valid: list, default: str) -> str:
    """Map a model-returned label onto one of `valid`, never raising.

    Models sometimes echo the prompt's option list ("action mixed dialogue
    reveal") or add casing/whitespace. Exact match wins; otherwise if the
    string contains exactly one valid option as a word, take it; else fall
    back to `default`. A bad label must never kill a run after tokens were
    spent generating the plan.
    """
    s = str(raw or "").strip()
    if s in valid:
        return s
    low = s.lower()
    for v in valid:
        if v.lower() == low:
            return v
    words = set(low.split())
    hits = [v for v in valid if v.lower() in words]
    if len(hits) == 1:
        return hits[0]
    return default


def _validate_beats(beats: list, runtime: float, target: int) -> list:
    prev_end = -1.0
    seen_types = set()
    kept: list = []
    for b in beats:
        for k in ("beat", "sceneType", "source_span", "visual_style", "summary"):
            if k not in b:
                raise ScriptError(f"beat missing {k!r}: {b}")
        coerced = _coerce_choice(b["sceneType"], SCENE_TYPES, "RISING")
        if coerced != b["sceneType"]:
            print(f"  warning: coerced sceneType {b['sceneType']!r} -> {coerced!r}")
        b["sceneType"] = coerced
        coerced = _coerce_choice(b["visual_style"], VISUAL_STYLES, "mixed")
        if coerced != b["visual_style"]:
            print(f"  warning: coerced visual_style {b['visual_style']!r} -> {coerced!r}")
        b["visual_style"] = coerced
        s, e = b["source_span"]
        s, e = float(s), float(e)
        # A beat starting past the film's end is hallucinated — the model
        # ran long doing beat arithmetic. Drop it, don't die: the beats
        # before it already cover the film up to here.
        if s >= runtime - 1:
            print(f"  warning: dropped {b['beat']} — "
                  f"span [{s:.0f},{e:.0f}] starts past film end ({runtime:.0f}s)")
            continue
        # Models overshoot the runtime or drift a little on boundaries.
        # Clamp into the film with a warning instead of killing the run
        # over a repairable slip — same rule as label coercion above.
        if e > runtime:
            print(f"  warning: clamped {b['beat']} end {e:.0f} -> {runtime:.0f}")
            e = runtime
        if s < 0:
            print(f"  warning: clamped {b['beat']} start {s:.0f} -> 0")
            s = 0.0
        if s < prev_end - 0.01:
            print(f"  warning: clamped {b['beat']} start {s:.0f} -> {prev_end:.0f} (overlap)")
            s = prev_end
        if not s < e:
            raise ScriptError(f"span [{s},{e}] unusable for film (0-{runtime:.0f})")
        prev_end = e
        seen_types.add(b["sceneType"])
        b["source_span"] = [round(s, 2), round(e, 2)]
        kept.append(b)
    if not kept:
        raise ScriptError("planner returned no beats inside the film")
    # The tail must be covered: if the last kept beat stops short of the
    # film's end, stretch it so no footage is left undescribed.
    if kept[-1]["source_span"][1] < runtime - 1:
        old = kept[-1]["source_span"][1]
        kept[-1]["source_span"][1] = round(runtime, 2)
        print(f"  warning: stretched {kept[-1]['beat']} end {old:.0f} -> {runtime:.0f} (tail cover)")
    beats = kept
    if abs(len(beats) - target) > max(2, target // 4):
        print(f"  warning: planner returned {len(beats)} beats, "
              f"asked for {target} — continuing")
    return beats


# ---------------------------------------------------------------- pass B: write

WRITE_SYSTEM = """You write Hindi voiceover for "movie explained" YouTube videos.
Devanagari script only. You are a storyteller, not a describer — the viewer
must feel a human dubbed this, not a machine summarizing clips.

GOLD STANDARD — match this voice exactly:
"ऑफिसर उस आदमी के पास पहुंचा जिसे सब टैंक मैन कहकर बुलाते थे। लेकिन उसका
असली नाम और रैंक किसी को नहीं पता। ऑफिसर को बताया गया कि अब तक यह आदमी सात
बार भागने की कोशिश कर चुका है कैद से। यह बात सुनकर नाजी ऑफिसर और भड़क गया।"

HARD RULES — violating any of these is failure:
1. PURE HINDI. The only English allowed is proper nouns (character names) and
   loanwords already natural in Hindi (टैंक, ऑफिसर, कमांडर). NEVER quote
   English dialogue, NEVER drop English phrases ("Don't move!", "backup",
   "International law", "Execute!"). If a line matters, render it in Hindi:
   उसने कहा कि मेरा सर सिर्फ मेरे देश के लिए झुकता है।
2. ONE CONNECTED STORY. Each beat must causally follow the previous one —
   इसलिए, फिर, लेकिन, तभी, इसके बाद. Never a standalone vignette; the viewer
   must always know why we are here now.
3. CONCRETE, never abstract. Names, numbers, specific actions. BANNED empty
   phrases: "तनाव चरम पर", "माहौल भारी", "दिल दहला देने वाला", "ट्विस्ट",
   "राइजिंग टेंशन". If something twists, SHOW the twist happening — don't
   announce it.
4. NO META-TALK. Never narrate the narration: no "टोन बदलता है", no "ब्रेक
   मिलता है", no "यहीं से ट्विस्ट". Never open a beat with "दोस्तों".
5. Past tense, third person, steady. One call-to-action ("कमेंट्स में जरूर
   बताना") at the very end of the final beat only — nowhere else.
6. GROUNDING: use names and events from the provided film dialogue; don't
   invent new ones. If the transcript is fragmentary, narrate only what is
   certain — never fill gaps with generic action.

For each beat you get: its summary, sceneType, and the film's real dialogue
inside its span (grounding — use names and events from it, don't invent new ones).

Reply with ONLY a JSON array, one object per beat in order:
{"beat": "BEAT_01",
 "voiceoverText": "Devanagari Hindi narration, 450-700 characters",
 "voiceMetadata": {"emotion": "curious|tense|somber|triumphant|conspiratorial|shocked",
                   "energy": 0.0-1.0, "pace": "slow|normal|fast",
                   "emphasis": ["2-4 key Hindi words"],
                   "pause_after_words": ["a phrase where a beat of silence lands"]}}

Rules: narration must tell what the summary says happens — the viewer sees
footage of exactly this. No commentary, no code fences, only the JSON array."""


def _span_dialogue(brief: dict, span: list) -> str:
    s, e = span
    parts = []
    for w in brief.get("windows") or []:
        if w.get("end", 0) >= s and w.get("start", 0) <= e:
            d = (w.get("dialogue") or "").strip()
            if d:
                parts.append(f"({w.get('timecode')}) {d[:300]}")
    return "\n".join(parts[:12])


def write_beats(beats: list, brief: dict) -> list:
    out = []
    for i in range(0, len(beats), WRITE_BATCH):
        batch = beats[i:i + WRITE_BATCH]
        items = []
        for b in batch:
            items.append(
                f"BEAT {b['beat']} [{b['sceneType']}]\n"
                f"summary: {b['summary']}\n"
                f"film dialogue in span:\n{_span_dialogue(brief, b['source_span'])}")
        user = ("Write Hindi voiceover for these beats:\n\n" +
                "\n\n".join(items))
        written = chat_json(
            [{"role": "system", "content": WRITE_SYSTEM},
             {"role": "user", "content": user}],
            temperature=0.75, max_tokens=8000)
        if not isinstance(written, list) or len(written) != len(batch):
            raise ScriptError(
                f"writer returned {len(written) if isinstance(written, list) else '?'} "
                f"beats for a batch of {len(batch)}")
        by_beat = {w.get("beat"): w for w in written}
        for b in batch:
            w = by_beat.get(b["beat"])
            if not w or not (w.get("voiceoverText") or "").strip():
                raise ScriptError(f"no narration for {b['beat']}")
            text = w["voiceoverText"].strip()
            if len(text) < 150:
                raise ScriptError(f"narration for {b['beat']} suspiciously "
                                  f"short ({len(text)} chars)")
            b["voiceoverText"] = text
            b["voiceMetadata"] = w.get("voiceMetadata") or {}
        out.extend(batch)
        print(f"  wrote {len(out)}/{len(beats)} beats "
              f"({sum(len(b['voiceoverText']) for b in out)} chars)")
    return out


# ---------------------------------------------------------------- pass C: judge

JUDGE_SYSTEM = """You are a ruthless YouTube editor judging a Hindi "movie explained" script.
The bar: a human dub, not AI slop. Score 1-10 on:
- hindi_purity: zero English phrases or quoted English (proper nouns only).
  Any "Don't move!", "backup", "International law", "Execute!" -> score <= 3.
- causality: every beat follows from the previous one; no standalone vignettes
  the viewer can't place.
- concreteness: names, numbers, specific actions. Empty critic-phrases like
  "तनाव चरम पर", "माहौल भारी", "दिल दहला देने वाला" -> low score.
- no_meta: no narration about the narration — no "टोन बदलता है", no "ब्रेक
  मिलता है", no beat opened with "दोस्तों", no announcing "ट्विस्ट".
- story_flow: does it build and pay off?
- hook_strength: first 60 seconds.
- pacing: no dead stretches.
Reply ONLY JSON: {"hindi_purity": N, "causality": N, "concreteness": N,
"no_meta": N, "story_flow": N, "hook_strength": N, "pacing": N,
"weak_beats": ["BEAT_04", ...], "notes": "one line naming the worst flaw"}"""


def judge_script(scenes: list) -> dict:
    body = "\n\n".join(
        f"{s['beat']} [{s['sceneType']}]: {(s.get('voiceoverText') or '')[:500]}"
        for s in scenes)
    return chat_json(
        [{"role": "system", "content": JUDGE_SYSTEM},
         {"role": "user",
          "content": f"Judge this script:\n\n{body}"}],
        temperature=0.3, max_tokens=1500)


REFINE_SYSTEM = """You rewrite weak beats of a Hindi "movie explained" script.
Fix exactly these flaws:
- English phrases/quotes -> render in pure Hindi (proper nouns only).
- Beat doesn't connect to the previous one -> add causal tissue (इसलिए/फिर/लेकिन/तभी).
- Abstract filler ("तनाव चरम पर", "माहौल भारी", "दिल दहला देने वाला") ->
  concrete names, numbers, specific actions.
- Meta-talk ("दोस्तों", "टोन बदलता है", "ट्विस्ट", "ब्रेक मिलता है") -> delete it,
  just tell the story.
You get each beat's current narration, its summary, and the judge's note.
Return ONLY a JSON array: [{"beat": "BEAT_04", "voiceoverText": "...",
"voiceMetadata": {...}}] — same schema, 450-700 characters, same hard rules."""


def refine_beats(scenes: list, weak: list, note: str) -> list:
    by_beat = {s["beat"]: s for s in scenes}
    items = []
    for label in weak:
        s = by_beat.get(label)
        if not s:
            continue
        items.append(f"BEAT {label} [{s['sceneType']}]\nsummary: {s['summary']}\n"
                     f"judge note: {note}\ncurrent: {s['voiceoverText']}")
    if not items:
        return scenes
    fixed = chat_json(
        [{"role": "system", "content": REFINE_SYSTEM},
         {"role": "user", "content": "Rewrite these beats:\n\n" +
          "\n\n".join(items)}],
        temperature=0.7, max_tokens=6000)
    by_fixed = {f.get("beat"): f for f in (fixed if isinstance(fixed, list) else [])}
    for s in scenes:
        f = by_fixed.get(s["beat"])
        if f and (f.get("voiceoverText") or "").strip():
            s["voiceoverText"] = f["voiceoverText"].strip()
            if f.get("voiceMetadata"):
                s["voiceMetadata"] = f["voiceMetadata"]
    return scenes


# ---------------------------------------------------------------- driver

def write_script(brief_path: str, *, out_path: str,
                 target_minutes: float = 25.0) -> dict:
    with open(brief_path, encoding="utf-8") as f:
        brief = json.load(f)
    runtime = float(brief.get("source_duration") or 0)
    target_chars = int(target_minutes * CHARS_PER_MIN)
    # ~600 chars/beat is the sweet spot: long enough to say something,
    # short enough that one TTS line stays directable.
    target_beats = max(8, min(48, round(target_chars / 600)))
    film = brief.get("source") or os.path.basename(brief_path)

    print(f"  film: {os.path.basename(str(film))} ({runtime / 60:.1f} min)")
    print(f"  target: ~{target_minutes:.0f} min narration "
          f"(~{target_chars} chars, ~{target_beats} beats)")

    print("  pass A: planning beats...")
    beats = plan_beats(brief, target_beats=target_beats)
    print(f"  planned {len(beats)} beats")

    print("  pass B: writing Hindi narration...")
    beats = write_beats(beats, brief)

    print("  pass C: judging...")
    judge = judge_script(beats)
    print(f"    hindi_purity={judge.get('hindi_purity')} "
          f"causality={judge.get('causality')} "
          f"concreteness={judge.get('concreteness')} "
          f"no_meta={judge.get('no_meta')} "
          f"story_flow={judge.get('story_flow')} "
          f"hook={judge.get('hook_strength')} "
          f"pacing={judge.get('pacing')}")
    refines = 0
    weak = judge.get("weak_beats") or []
    scores = [judge.get(k, 10) for k in
              ("hindi_purity", "causality", "concreteness", "no_meta",
               "story_flow", "hook_strength", "pacing")]
    while weak and min(scores) < JUDGE_BAR and refines < MAX_REFINES:
        refines += 1
        print(f"  refine {refines}: rewriting {len(weak)} weak beats...")
        beats = refine_beats(beats, weak, judge.get("notes", ""))
        judge = judge_script(beats)
        weak = judge.get("weak_beats") or []
        scores = [judge.get(k, 10) for k in
                  ("hindi_purity", "causality", "concreteness", "no_meta",
                   "story_flow", "hook_strength", "pacing")]
        print(f"    after refine: min score {min(scores)}")

    total_chars = sum(len(b["voiceoverText"]) for b in beats)
    script = {
        "title_working": f"explained: {os.path.basename(str(film))}",
        "language": "hi",
        "film": film,
        "runtime_s": runtime,
        "target_minutes": target_minutes,
        "estimated_minutes": round(total_chars / CHARS_PER_MIN, 1),
        "beats_planned": len(beats),
        "judge": judge,
        "refines": refines,
        "scenes": beats,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(script, f, indent=2, ensure_ascii=False)
    print(f"  script -> {out_path} "
          f"({total_chars} chars, ~{script['estimated_minutes']} min)")
    return script


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2 or args[1] in ("-h", "--help"):
        print(__doc__)
        print("usage:")
        print("  python src/script_writer.py <brief.json> [--out script.json] "
              "[--minutes 25]")
        print("  env: POLLINATIONS_API_KEY, POLLINATIONS_MODEL (default openai)")
        sys.exit(0)
    brief_path = args[1]
    out = "script.json"
    minutes = 25.0
    if "--out" in args:
        out = args[args.index("--out") + 1]
    if "--minutes" in args:
        minutes = float(args[args.index("--minutes") + 1])
    print("=" * 68)
    print("  SCRIPT WRITER")
    print("=" * 68)
    write_script(brief_path, out_path=out, target_minutes=minutes)
