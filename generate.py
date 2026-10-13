#!/usr/bin/env python
"""Generate one song with ACE-Step 1.5 XL-SFT and write it to a file — the AI Runtime batch job's entry point.

It prepares the pinned source + checkpoints (acestep_core.prepare), loads the serving PyFunc in-process and sends one
`generate` request, so the parameters are exactly the endpoint's (docs/SERVING-API.md). Needs a CUDA GPU and the
runtime from scripts/install.sh.

    python generate.py --caption "cinematic orchestral trailer, epic drums, choir" --lyrics "[Instrumental]" \\
        --duration 30 --seed 42 --out /tmp/acestep_out/song.mp3
    python generate.py --params '{"caption": "...", "lyrics": "...", "duration": 60, "bpm": 96}' --out song.flac
"""
import argparse
import base64
import json
from pathlib import Path

import acestep_core as core


def main() -> None:
    ap = argparse.ArgumentParser(description="ACE-Step 1.5 XL-SFT song generation")
    ap.add_argument("--caption", help="music description (genre, instruments, mood ...)")
    ap.add_argument("--lyrics", default="[Instrumental]", help="lyrics text, or a path to a .txt file")
    ap.add_argument("--duration", type=float, default=30.0, help="seconds")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--params", default="", help="a JSON object of any generate parameters; overrides the flags")
    ap.add_argument("--out", required=True, help="output path; the extension sets audio_format (mp3 / flac / wav ...)")
    ap.add_argument("--work", default="/tmp/acestep", help="where the source and checkpoints are prepared")
    args = ap.parse_args()

    out = Path(args.out)
    lyrics = Path(args.lyrics).read_text(encoding="utf-8") if Path(args.lyrics).is_file() else args.lyrics
    params = {"caption": args.caption, "lyrics": lyrics, "duration": args.duration, "seed": args.seed,
              "audio_format": out.suffix.lstrip(".") or "mp3"}
    params.update(json.loads(args.params) if args.params else {})
    if not params.get("caption"):
        raise SystemExit("--caption (or caption in --params) is required")

    _, ckpt = core.prepare(args.work)
    model = core.load_model(ckpt)
    res = core.run(model, "generate", params)
    a = res["audios"][0]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(base64.b64decode(a["audio_b64"]))
    (out.with_suffix(".json")).write_text(json.dumps({k: v for k, v in res.items() if k != "audios"} |
                                                     {"audio": {k: v for k, v in a.items() if k != "audio_b64"}},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
    core.log(f"wrote {out} ({a['format']}, {a['sample_rate']} Hz, {a['seconds']} s) in {res.get('wall_seconds')} s")


if __name__ == "__main__":
    main()
