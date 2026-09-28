#!/usr/bin/env python3
"""
pipeline.py — the whole movie-explainer, one command.

    python pipeline.py "D:\\movies\\film.mp4" --minutes 25

Stages (each resumable with --resume; a stage whose outputs already exist
is skipped):
  1. sourceRead   film -> brief.json            (shots + transcript beat-sheet)
  2. script_writer brief -> script.json         (Hindi narration, span-tagged)
  3. voiceover    script -> manifest.json       (local TTS + timing manifest)
  4. aligner      script+manifest -> edl.json   (clip/freeze-frame edit list)
  5. assemble     edl+manifest -> film_explained.mp4
  6. metadata     script+brief -> metadata.txt  (title/desc/keywords/thumbnail prompt)

Everything lands in --out (default: output/<film-stem>/).
No uploads happen here — Dad uploads the finished file manually.

Env (.env next to this file): POLLINATIONS_API_KEY, POLLINATIONS_MODEL,
optional TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID for a "done" ping.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
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


HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
DATA_DIR = os.path.join(SRC, "data")
WORK_DIR = os.path.join(SRC, "source_work")


def load_env():
    path = os.path.join(HERE, ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def run(cmd, *, cwd=HERE):
    print(f"  $ {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-3000:])
        print(p.stderr[-3000:])
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd[:3])}")
    tail = (p.stdout or "").strip().splitlines()
    for line in tail[-4:]:
        print(f"    {line}")
    return p


def check_prereqs(need_whisper: bool):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg/ffprobe not on PATH")
    if not os.getenv("POLLINATIONS_API_KEY"):
        raise RuntimeError("POLLINATIONS_API_KEY missing — set it in .env")
    try:
        import requests  # noqa
    except ImportError:
        raise RuntimeError("pip install requests")
    if need_whisper:
        try:
            import faster_whisper  # noqa
        except ImportError:
            raise RuntimeError("pip install faster-whisper  (needed for stage 1)")


def notify(text: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    helper = os.path.join(HERE, "sendTelegramNotification.py")
    if not (token and chat_id and os.path.exists(helper)):
        return
    try:
        subprocess.run([sys.executable, helper, text],
                       capture_output=True, timeout=30)
    except Exception:
        pass


def main() -> int:
    load_env()
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    film = os.path.abspath(args[0])
    if not os.path.exists(film):
        print(f"no such file: {film}")
        return 1
    stem = os.path.splitext(os.path.basename(film))[0]
    out = os.path.join(HERE, "output", stem)
    minutes = 25.0
    resume = "--resume" in args
    if "--out" in args:
        out = os.path.abspath(args[args.index("--out") + 1])
    if "--minutes" in args:
        minutes = float(args[args.index("--minutes") + 1])
    os.makedirs(out, exist_ok=True)

    brief_out = os.path.join(out, "brief.json")
    script_out = os.path.join(out, "script.json")
    manifest_out = os.path.join(out, "manifest.json")
    plan_out = os.path.join(out, "plan.json")
    edl_out = os.path.join(out, "edl.json")
    final_out = os.path.join(out, f"{stem}_explained.mp4")

    skip1 = resume and os.path.exists(brief_out)
    check_prereqs(need_whisper=not skip1)
    t0 = time.time()
    print("=" * 68)
    print(f"  MOVIE EXPLAINER  —  {os.path.basename(film)}")
    print(f"  out: {out}   target: ~{minutes:.0f} min narration")
    print("=" * 68)

    def stage(n, name, done, fn):
        if done:
            print(f"\n[{n}/6] {name} — skipped (output exists, --resume)")
            return
        print(f"\n[{n}/6] {name}")
        s = time.time()
        fn()
        print(f"  done in {(time.time() - s) / 60:.1f} min")

    # -- 1. brief ---------------------------------------------------------
    def s1():
        run([sys.executable, os.path.join(SRC, "sourceRead.py"), film])
        src = os.path.join(DATA_DIR, f"source_brief_{stem}.json")
        if not os.path.exists(src):
            raise RuntimeError(f"sourceRead did not write {src}")
        shutil.copy(src, brief_out)
    stage(1, "sourceRead — shots + transcript", resume and os.path.exists(brief_out), s1)

    # -- 2. script --------------------------------------------------------
    def s2():
        run([sys.executable, os.path.join(SRC, "script_writer.py"),
             brief_out, "--out", script_out, "--minutes", str(minutes)])
    stage(2, "script_writer — Hindi narration", resume and os.path.exists(script_out), s2)

    # -- 3. voiceover -----------------------------------------------------
    def s3():
        run([sys.executable, os.path.join(SRC, "voiceover.py"),
             "--script", script_out])
        src = os.path.join(DATA_DIR, "voiceover_manifest.json")
        if not os.path.exists(src):
            raise RuntimeError("voiceover did not write its manifest")
        shutil.copy(src, manifest_out)
    stage(3, "voiceover — local Hindi TTS", resume and os.path.exists(manifest_out), s3)

    with open(manifest_out, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(script_out, encoding="utf-8") as f:
        script = json.load(f)
    narration_min = sum(t.get("duration", 0) for t in manifest.get("timeline", [])) / 60

    # -- 4. align ---------------------------------------------------------
    def s4():
        scenes, tl = script["scenes"], manifest["timeline"]
        if len(scenes) != len(tl):
            raise RuntimeError(
                f"script has {len(scenes)} beats but manifest has {len(tl)} — "
                "refusing to guess the mapping")
        beats = []
        for sc, t in zip(scenes, tl):
            label = sc.get("beat") or sc.get("sceneType")
            if t.get("beat") != label:
                print(f"  warning: manifest beat {t.get('beat')!r} != "
                      f"script {label!r} (positional merge)")
            beats.append({"beat": label,
                          "duration": float(t["duration"]),
                          "source_span": sc["source_span"],
                          "visual_style": sc["visual_style"]})
        # script["film"] is the brief's media-info dict; the aligner needs a path
        film_field = script.get("film") or film
        if isinstance(film_field, dict):
            film_field = film_field.get("path") or film
        plan = {"source": os.path.abspath(film_field),
                "beats": beats}
        with open(plan_out, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
        run([sys.executable, os.path.join(SRC, "aligner.py"),
             plan_out, "--work", WORK_DIR, "--out", edl_out])
    stage(4, "aligner — clip/freeze edit list", resume and os.path.exists(edl_out), s4)

    # -- 5. assemble ------------------------------------------------------
    def s5():
        run([sys.executable, os.path.join(SRC, "assemble.py"),
             edl_out, manifest_out, "--out", final_out])
    stage(5, "assemble — final render", resume and os.path.exists(final_out), s5)

    # -- 6. metadata ------------------------------------------------------
    def s6():
        run([sys.executable, os.path.join(SRC, "metadata.py"),
             script_out, brief_out, "--out", out,
             "--minutes", f"{narration_min:.1f}"])
    stage(6, "metadata — title/desc/keywords/thumbnail prompt",
          resume and os.path.exists(os.path.join(out, "metadata.txt")), s6)

    hrs = (time.time() - t0) / 3600
    print("\n" + "=" * 68)
    print(f"  DONE in {hrs:.1f}h")
    print(f"  video    : {final_out}")
    print(f"  narration: ~{narration_min:.1f} min")
    print(f"  metadata : {os.path.join(out, 'metadata.txt')}")
    print("=" * 68)
    notify(f"Explainer done: {os.path.basename(final_out)} "
           f"(~{narration_min:.0f} min, {hrs:.1f}h processing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
