"""
metadata.py — title, description, keywords and the thumbnail prompt.

Pipeline position:
    sourceRead -> script_writer -> voiceover -> aligner -> assemble -> metadata

One Pollinations call turns the finished script (+ the film's own brief) into
upload-ready packaging. The thumbnail itself is NOT generated here — the
thumbnail PROMPT goes into metadata.txt, and Dad renders it wherever he likes.

Output: metadata.json (machine) + metadata.txt (human, copy-paste ready).

Thumbnail-prompt convention: English, not Hindi — image models render Devanagari
unreliably. When people appear, they read as Indian, with clothing and setting
matched to the film.
"""

from __future__ import annotations

import json
import os
import sys
import time

from script_writer import chat_json, ScriptError

# Windows consoles default to cp1252, which crashes printing non-ASCII
# (Devanagari narration, transcript dialogue). Replace unprintable chars
# with ? instead of dying mid-run.
try:
    if sys.stdout is not None:
        sys.stdout.reconfigure(errors="replace")
except Exception:
    pass


META_SYSTEM = """You package Hindi "movie explained" YouTube videos.
You get the narration script (Hindi) and the film's beat-sheet.
Reply with ONLY JSON:
{"title": "Hindi title, <=70 chars, curiosity + the film's hook, no clickbait lies",
 "description": "Hindi, 2-3 short paragraphs: what the video covers, why this film is worth the recap, then 1 line inviting comments",
 "keywords": ["12-18 Hindi/English search terms, film name variants included"],
 "hashtags": ["#MovieExplained", ...up to 5],
 "thumbnail_prompt": "English, one vivid paragraph for an image model: single dramatic moment, expressive Indian faces, bold composition, space for 2-3 word Hindi text overlay",
 "thumbnail_text": "2-3 Hindi words for the thumbnail overlay",
 "chapters": [{"t": "00:00", "label": "Hindi chapter label"} ... 6-10 chapters across the narration runtime]}

Rules: title/description/chapters in Hindi (Devanagari). Keywords mix Hindi and
English the way Indian viewers actually search. Never spoil the final twist in
the title. No commentary, only the JSON."""


def build_metadata(script_path: str, brief_path: str, *,
                   out_dir: str, narration_minutes: float) -> dict:
    with open(script_path, encoding="utf-8") as f:
        script = json.load(f)
    with open(brief_path, encoding="utf-8") as f:
        brief = json.load(f)

    scenes_txt = "\n".join(
        f"{s['beat']} [{s['sceneType']}]: {(s.get('voiceoverText') or '')[:280]}"
        for s in script.get("scenes", []))
    film = os.path.basename(str(script.get("film") or brief.get("source") or "film"))
    user = (f"Film file: {film}\n"
            f"Narration runtime: ~{narration_minutes:.0f} minutes\n"
            f"Judge scores: {json.dumps(script.get('judge', {}), ensure_ascii=False)}\n\n"
            f"SCRIPT:\n{scenes_txt}")

    print("  generating title/description/keywords/thumbnail prompt...")
    meta = chat_json(
        [{"role": "system", "content": META_SYSTEM},
         {"role": "user", "content": user}],
        temperature=0.7, max_tokens=4000)
    if not isinstance(meta, dict) or "title" not in meta:
        raise ScriptError("metadata model did not return a usable object")

    meta["film"] = film
    meta["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metadata.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    chapters = "\n".join(
        f"  {c.get('t', '')}  {c.get('label', '')}"
        for c in (meta.get("chapters") or []))
    txt = f"""TITLE
{meta.get('title', '')}

DESCRIPTION
{meta.get('description', '')}

KEYWORDS
{', '.join(meta.get('keywords') or [])}

HASHTAGS
{' '.join(meta.get('hashtags') or [])}

CHAPTERS
{chapters}

THUMBNAIL TEXT OVERLAY
{meta.get('thumbnail_text', '')}

THUMBNAIL PROMPT (paste into your image generator)
{meta.get('thumbnail_prompt', '')}
"""
    with open(os.path.join(out_dir, "metadata.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    print(f"  metadata -> {out_dir}/metadata.txt + metadata.json")
    return meta


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 3 or args[1] in ("-h", "--help"):
        print(__doc__)
        print("usage:")
        print("  python src/metadata.py <script.json> <brief.json> "
              "--out <dir> [--minutes 25]")
        sys.exit(0)
    out = "."
    minutes = 25.0
    if "--out" in args:
        out = args[args.index("--out") + 1]
    if "--minutes" in args:
        minutes = float(args[args.index("--minutes") + 1])
    print("=" * 68)
    print("  METADATA")
    print("=" * 68)
    build_metadata(args[1], args[2], out_dir=out, narration_minutes=minutes)
