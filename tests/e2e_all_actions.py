#!/usr/bin/env python3
"""End-to-end test of every action of the ACE-Step endpoint (sequential — the endpoint has one GPU).

    export DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com DATABRICKS_CONFIG_PROFILE=<profile>
    python tests/e2e_all_actions.py --src-audio path/to/any_song.wav [--only status,generate_t2m,...]

`--src-audio` is any music file (≥ ~40 s); a 20 s excerpt of it is used for the audio-to-audio actions.
Outputs go to tests/out/, plus tests/out/summary.json.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "client"))
from acestep_client import AceStep  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)

TESTS = {
    "status": ("status", None, {}),
    "list_params": ("list_params", None, {}),
    "create_sample": ("create_sample", None,
                      {"query": "a mysterious epic sci-fi trailer theme about two worlds joined by an ancient gate",
                       "instrumental": True}),
    "format_sample": ("format_sample", None,
                      {"caption": "j-pop ballad, female vocal, piano",
                       "lyrics": "[verse]\n夜の窓に 光が揺れる\n[chorus]\n君を呼ぶ 風の中で"}),
    "generate_t2m": ("generate", None,
                     {"caption": "epic hybrid orchestral trailer music, taiko, choir, deep braams, dark to triumphant",
                      "lyrics": "[Instrumental]", "duration": 30, "seed": 7, "audio_format": "mp3"}),
    "generate_vocal_ja_batch2": ("generate", None,
                                 {"caption": "japanese city pop, female vocal, warm electric piano, slap bass, 1980s",
                                  "vocal_language": "ja",
                                  "lyrics": "[verse]\n夜の窓に 光が揺れる\n君の声が 遠くで響く\n[chorus]\nつないだ手を 離さないで",
                                  "duration": 40, "bpm": 104, "keyscale": "A minor", "seed": 11, "batch_size": 2,
                                  "audio_format": "flac"}),
    "generate_no_lm_advanced": ("generate", None,
                                {"caption": "lofi hip hop, mellow piano, vinyl crackle", "lyrics": "[Instrumental]",
                                 "duration": 20, "thinking": False, "use_cot_metas": False, "use_cot_caption": False,
                                 "use_cot_language": False, "inference_steps": 30, "guidance_scale": 5.0, "shift": 3.0,
                                 "sampler_mode": "heun", "infer_method": "ode", "fade_in_duration": 1.0,
                                 "fade_out_duration": 2.0, "seed": 3, "audio_format": "wav"}),
    "audio_to_codes": ("audio_to_codes", "SRC", {}),
    "understand": ("understand", "SRC", {}),
    "cover": ("generate", "SRC", {"task_type": "cover", "caption": "solo jazz piano trio cover, brushes, upright bass",
                                  "audio_cover_strength": 0.6, "lyrics": "[Instrumental]", "seed": 5,
                                  "audio_format": "mp3"}),
    "repaint": ("generate", "SRC", {"task_type": "repaint", "repainting_start": 6.0, "repainting_end": 12.0,
                                    "chunk_mask_mode": "explicit", "caption": "a sudden quiet breakdown with a solo cello",
                                    "lyrics": "[Instrumental]", "seed": 9, "audio_format": "mp3"}),
    "reject_base_only_task": ("generate", "SRC", {"task_type": "extract",
                                                  "instruction": "Extract the drums track from the audio:"}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--src-audio", default="")
    a = ap.parse_args()
    names = [x for x in a.only.split(",") if x] or list(TESTS)
    src = None
    if any(TESTS[n][1] == "SRC" for n in names):
        if not a.src_audio:
            sys.exit("--src-audio is required for audio-to-audio tests")
        src = str(OUT / "_src_excerpt.mp3")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "20", "-t", "20", "-i", a.src_audio, "-b:a", "192k", src],
                       check=True)
    ace = AceStep(); summary = {}
    for n in names:
        action, s, params = TESTS[n]
        t0 = time.time()
        src_b64, src_ext = ace._b64(src if s == "SRC" else None)
        res, err = ace.call({"action": action, "params_json": json.dumps(params, ensure_ascii=False),
                             "src_audio_b64": src_b64, "src_audio_ext": src_ext})
        files = ace.save_audios(res, str(OUT / n)) if res.get("audios") else []
        brief = {k: v for k, v in res.items() if k not in ("audios", "audio_codes", "generate", "applied")}
        if res.get("audios"):
            brief["audios"] = [{k: v for k, v in x.items() if k in ("format", "seconds", "seed", "sample_rate")}
                               for x in res["audios"]]
        if "audio_codes" in res:
            brief["audio_codes_prefix"] = res["audio_codes"][:80]
        summary[n] = {"wall_s": round(time.time() - t0, 1), "error": err, "files": files, "result": brief}
        print(f"== {n} ({summary[n]['wall_s']}s) error={err!r}\n{json.dumps(brief, ensure_ascii=False)[:1000]}", flush=True)
        json.dump(summary, open(OUT / "summary.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
