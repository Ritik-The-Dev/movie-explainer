# Movie Explainer — local Hindi "movie explained" pipeline

One command turns a downloaded film into a 20–30 minute Hindi explainer:
narration matched to the exact scenes, sub-5-second live clips, Ken Burns
freeze-frames, polished local TTS, and upload-ready metadata.

```
film.mp4
  │ 1. sourceRead    shots + transcript beat-sheet (faster-whisper, CPU int8)
  ▼
brief.json ──► 2. script_writer  Hindi narration via Pollinations, every beat
  │              tagged with source_span + visual_style, judge-and-refine pass
  ▼
script.json ─► 3. voiceover     local TTS (Voicebox → Cartesia → Kokoro),
  │              per-beat timing manifest, de-glitch polish chain
  ▼
manifest.json + script.json ──► 4. aligner   clip (<5s) / freeze-frame EDL
  │              from the script's own source_spans — inspectable before render
  ▼
edl.json ─────► 5. assemble     pure-ffmpeg 1080p30 render, narration-locked
  │
  └───────────► 6. metadata      title, description, keywords, hashtags,
                                 chapters + thumbnail prompt → metadata.txt
```

## Quick start (Windows)

```bat
cd movie-explainer
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
notepad .env   :: set POLLINATIONS_API_KEY
```

Verify the toolchain:

```bat
python src\sourceRead.py --check        :: ffmpeg + whisper deps
python src\voiceover.py --bench         :: settle TTS speed (~5 min, one time)
python src\voiceover.py --say "दोस्तों, यह एक टेस्ट है"
```

Run a film (queue 3–4 at night, ~3h each):

```bat
python pipeline.py "D:\movies\film.mp4" --minutes 25
```

Output lands in `output\<film>\`: `film_explained.mp4`, `metadata.txt`
(copy-paste into YouTube), plus `brief.json`, `script.json`, `manifest.json`,
`plan.json`, `edl.json` for inspection. Resume an interrupted run with
`--resume` — finished stages are skipped.

## Stage notes

- **script_writer** is the quality gate. It plans beats against the brief's
  windows, writes Hindi narration grounded in the film's real dialogue,
  then a judge pass rewrites weak beats (once, bounded). Every beat carries
  `source_span` in absolute film seconds — aligner refuses to render a beat
  whose span fails validation.
- **voiceover** never trusts TTS blindly: silence-analysis measures each
  beat's real duration, the polish chain (EQ → de-esser → compressor →
  limiter → loudness) removes the jitter/glitch artifacts, and per-beat
  caching means a re-run only re-renders what changed.
- **aligner** prints its whole edit decision list before assemble runs —
  read `edl.json` if a visual ever feels off; the `reason` field says why
  each segment is a clip or a freeze.
- **metadata** writes `metadata.txt` with the title, description, keywords
  and the **thumbnail prompt** (render the thumbnail in your image tool of
  choice — prompts are English because image models mangle Devanagari).

## Hardware

Built for Ryzen 5 + RX 6500M (no CUDA) + 16 GB RAM: whisper runs int8 on CPU
in 20-minute chunks, TTS is local, all video work is ffmpeg subprocesses.
No torch, no OpenCV, no MoviePy — nothing that can fight your pinned numpy.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `POLLINATIONS_API_KEY is not set` | fill it in `.env` |
| voiceover engine slow first run | normal — models warm up; `--bench` once |
| narration feels rushed | raise `--minutes`, or lower `SCRIPT_CHARS_PER_MIN` |
| a visual mismatches the narration | check `edl.json` → the beat's `source_span` in `script.json` → fix and re-run from stage 4 with `--resume` |
| TTS glitch on one beat | delete that beat's file in `src/voice_out/` and re-run stage 3 with `--resume` |
