"""Tiny client for the ACE-Step Databricks Model Serving endpoint.

Configuration (environment variables):
    DATABRICKS_HOST              workspace URL, e.g. https://<your-workspace>.cloud.databricks.com
    DATABRICKS_CONFIG_PROFILE    Databricks CLI profile used to mint an OAuth token (default: DEFAULT)
    DATABRICKS_TOKEN             optional: a PAT/OAuth token to use instead of the CLI
    ACESTEP_ENDPOINT             serving endpoint name (default: acestep-v15-xl-sft)

    from acestep_client import AceStep
    ace = AceStep()
    res = ace.generate(caption="lofi hip hop", lyrics="[Instrumental]", duration=30, audio_format="mp3")
    ace.save_audios(res, "out/lofi")
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class AceStep:
    def __init__(self, host: Optional[str] = None, endpoint: Optional[str] = None, profile: Optional[str] = None,
                 timeout: int = 900, retries: int = 40):
        self.host = (host or os.environ.get("DATABRICKS_HOST", "")).rstrip("/")
        if not self.host:
            raise ValueError("set DATABRICKS_HOST (https://<your-workspace>.cloud.databricks.com)")
        self.endpoint = endpoint or os.environ.get("ACESTEP_ENDPOINT", "acestep-v15-xl-sft")
        self.profile = profile or os.environ.get("DATABRICKS_CONFIG_PROFILE", "DEFAULT")
        self.timeout, self.retries = timeout, retries

    # ------------------------------------------------------------------ transport
    def _token(self) -> str:
        if os.environ.get("DATABRICKS_TOKEN"):
            return os.environ["DATABRICKS_TOKEN"]
        out = subprocess.check_output(["databricks", "auth", "token", "--profile", self.profile])
        return json.loads(out)["access_token"]

    def call(self, row: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """POST one request row; returns (result_dict, error_string). Waits through scale-from-zero."""
        body = json.dumps({"dataframe_records": [row]}).encode()
        for _ in range(self.retries):
            req = urllib.request.Request(f"{self.host}/serving-endpoints/{self.endpoint}/invocations", body,
                                         {"Authorization": f"Bearer {self._token()}",   # fresh token each try
                                          "Content-Type": "application/json"})
            try:
                p = json.load(urllib.request.urlopen(req, timeout=self.timeout))["predictions"][0]
                return json.loads(p["result_json"] or "{}"), p["error"]
            except urllib.error.HTTPError as e:
                msg = e.read().decode()[:400]
                if e.code in (429, 503, 504, 555) or "timed out" in msg:   # cold start / busy
                    time.sleep(60); continue
                return {}, f"HTTP {e.code}: {msg}"
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(30)
        return {}, "gave up after retries (endpoint still scaling from zero?)"

    # ------------------------------------------------------------------ actions
    @staticmethod
    def _b64(path: Optional[str]) -> Tuple[str, str]:
        if not path:
            return "", ""
        p = Path(path)
        return base64.b64encode(p.read_bytes()).decode(), p.suffix.lstrip(".").lower() or "wav"

    def run(self, action: str, src_audio: Optional[str] = None, reference_audio: Optional[str] = None,
            **params) -> Dict[str, Any]:
        src_b64, src_ext = self._b64(src_audio)
        ref_b64, ref_ext = self._b64(reference_audio)
        row = {"action": action, "params_json": json.dumps(params, ensure_ascii=False),
               "src_audio_b64": src_b64, "src_audio_ext": src_ext,
               "reference_audio_b64": ref_b64, "reference_audio_ext": ref_ext}
        res, err = self.call(row)
        if err:
            raise RuntimeError(err)
        return res

    def generate(self, **kw):        return self.run("generate", **kw)
    def understand(self, **kw):      return self.run("understand", **kw)
    def create_sample(self, **kw):   return self.run("create_sample", **kw)
    def format_sample(self, **kw):   return self.run("format_sample", **kw)
    def audio_to_codes(self, src_audio): return self.run("audio_to_codes", src_audio=src_audio)
    def list_params(self):           return self.run("list_params")
    def status(self):                return self.run("status")

    @staticmethod
    def save_audios(res: Dict[str, Any], stem: str):
        out = []
        for i, a in enumerate(res.get("audios", [])):
            ext = "wav" if a["format"] == "wav32" else a["format"]
            p = Path(f"{stem}_{i}.{ext}"); p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(base64.b64decode(a["audio_b64"])); out.append(str(p))
        return out
