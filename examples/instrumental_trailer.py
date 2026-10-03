"""Instrumental cinematic trailer cue (68 s), two seeds.

    python examples/instrumental_trailer.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "client"))
from acestep_client import AceStep

STYLE = ("epic cinematic trailer music, instrumental, no vocals, hybrid orchestral, deep braams, taiko drums, "
         "rising strings ostinato, choir pads, sci-fi synth pulses, dark to triumphant build, huge climax")
ace = AceStep()
for seed in (202, 203):
    res = ace.generate(caption=STYLE, lyrics="[Instrumental]", instrumental=True, duration=68, seed=seed,
                       audio_format="wav")
    print(ace.save_audios(res, f"out/trailer_seed{seed}"))
