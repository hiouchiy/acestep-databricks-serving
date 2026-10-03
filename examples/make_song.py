"""Generate a full song with vocals (Japanese) — the example used to validate the endpoint.

    export DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com DATABRICKS_CONFIG_PROFILE=<profile>
    python examples/make_song.py
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "client"))
from acestep_client import AceStep

ace = AceStep()
print("status:", json.dumps(ace.status(), ensure_ascii=False)[:300])   # also wakes a scaled-to-zero endpoint
res = ace.generate(
    caption=("cinematic japanese pop rock anthem, female lead vocal, soaring chorus, driving drums, electric guitars, "
             "warm synth pads, strings, uplifting and emotional, wide stereo, polished mix"),
    lyrics=("[verse]\n千年の夜を越えて\n砂の街に風が吹く\n遠く空の向こうから\nだれかの歌が聞こえた\n\n"
            "[pre-chorus]\nほどけた糸をたぐれば\n胸の奥で光が鳴る\n\n"
            "[chorus]\nひらけ ゲートよ\nバラバラの世界をつないで\nこの手が届くなら\nもう一度 ひとつになれる\n\n"
            "[outro]\nずっと 待っていた\nすべてが つながるのを"),
    vocal_language="ja", duration=90, bpm=128, keyscale="D major", timesignature="4", seed=2026,
    audio_format="mp3", mp3_bitrate="320k")
print(ace.save_audios(res, "out/song"), res["time_costs"])
