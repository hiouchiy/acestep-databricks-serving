# Databricks notebook source
# MAGIC %md
# MAGIC # ACE-Step 1.5 XL-SFT — generate a song on serverless GPU
# MAGIC
# MAGIC Attach this notebook to **serverless GPU compute, accelerator A10** (environment version 4). Open it from a
# MAGIC Git folder (or a synced copy) of this repo, so `../acestep_core.py`, `../serving/` and `../scripts/install.sh`
# MAGIC are next to it.
# MAGIC
# MAGIC 1. Fill in the widgets. `params_json` takes **any** `generate` parameter of the endpoint (docs/SERVING-API.md):
# MAGIC    caption, lyrics, duration, bpm, keyscale, vocal_language, seed, audio_format, ...
# MAGIC 2. Run all. The first run installs the runtime (torch 2.10/cu128) and downloads ~36 GB of checkpoints
# MAGIC    (~20 min); a 90 s song then takes ~50 s.
# MAGIC
# MAGIC The engine is the serving PyFunc itself (`serving/acestep_pyfunc.py`), loaded in-process by `acestep_core`.

# COMMAND ----------

dbutils.widgets.text("params_json", '{"caption": "warm city-pop, female vocal, electric piano, slap bass, bright", '
                                    '"lyrics": "[verse]\\nNeon on the river\\n[chorus]\\nStay with me tonight", '
                                    '"duration": 30, "seed": 42, "audio_format": "mp3"}', "params_json")
dbutils.widgets.text("output_dir", "/Volumes/<CATALOG>/<SCHEMA>/<VOLUME>/acestep", "output_dir (UC Volume)")
dbutils.widgets.text("work_dir", "/tmp/acestep", "work_dir (source + checkpoints)")

# COMMAND ----------

# Install the runtime into THIS notebook's Python (same recipe as the AIR job), then restart Python.
import os
import subprocess
import sys

REPO = os.path.abspath("..")
subprocess.run(["bash", os.path.join(REPO, "scripts", "install.sh")], check=True,
               env={**os.environ, "PYTHON": sys.executable})
dbutils.library.restartPython()

# COMMAND ----------

import base64
import json
import os
import sys

REPO = os.path.abspath("..")
sys.path.insert(0, REPO)
import acestep_core as core  # noqa: E402

params = json.loads(dbutils.widgets.get("params_json"))
out_dir = dbutils.widgets.get("output_dir")
assert "<" not in out_dir, "set output_dir to a UC Volume folder, e.g. /Volumes/main/default/music/acestep"

_, ckpt = core.prepare(dbutils.widgets.get("work_dir"))
model = core.load_model(ckpt)
res = core.run(model, "generate", params)
a = res["audios"][0]
core.log(f"{a['format']} {a['sample_rate']} Hz {a['seconds']} s, wall {res.get('wall_seconds')} s")

# COMMAND ----------

import time  # noqa: E402

os.makedirs(out_dir, exist_ok=True)
name = f"{int(time.time())}.{a['format']}"
data = base64.b64decode(a["audio_b64"])
dest = os.path.join(out_dir, name)
open(dest, "wb").write(data)
open(os.path.join(out_dir, name + ".json"), "w").write(
    json.dumps({k: v for k, v in res.items() if k != "audios"} | {"audio": {k: v for k, v in a.items() if k != "audio_b64"}},
               ensure_ascii=False, indent=1))
print("wrote", dest)

from IPython.display import Audio, display  # noqa: E402
display(Audio(data=data))

# COMMAND ----------

import torch  # noqa: E402
dbutils.notebook.exit(json.dumps({"file": dest, "seconds": a["seconds"], "wall_seconds": res.get("wall_seconds"),
                                  "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}))
