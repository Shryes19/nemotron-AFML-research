"""OpenAI-compatible chat client for the shared Qwen2.5-7B-Instruct vLLM server.

Settings come only from LLM_BASE_URL, LLM_MODEL, LLM_API_KEY (environment variables, or a
.env file in the repo root). Responses are cached in sqlite keyed by a sha256 of
model + messages + sampling params, so reruns never re-query. Failed calls are retried with
exponential backoff. Every call is logged to cache/llm_calls.jsonl (latency, cache hit,
token usage) for compute accounting.

  python -m wm.llm_client --ping     # check the server is reachable
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
from typing import Dict, List, Optional

import requests

DEFAULT_CACHE = "cache/llm_cache.sqlite"
DEFAULT_LOG = "cache/llm_calls.jsonl"


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env reader; real environment variables take precedence."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def cache_key(model: str, messages: List[Dict], params: Dict) -> str:
    blob = json.dumps({"model": model, "messages": messages, "params": params}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class LLMClient:
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 api_key: Optional[str] = None, cache_path: str = DEFAULT_CACHE,
                 log_path: str = DEFAULT_LOG, max_retries: int = 5, backoff: float = 2.0,
                 timeout: float = 120.0):
        load_dotenv()
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "")).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "") or "Qwen/Qwen2.5-7B-Instruct"
        self.api_key = api_key if api_key is not None else os.environ.get("LLM_API_KEY", "")
        self.max_retries, self.backoff, self.timeout = max_retries, backoff, timeout
        self.log_path = log_path
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        self.db = sqlite3.connect(cache_path)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT, created REAL)")
        self.db.commit()

    def chat(self, messages: List[Dict], temperature: float = 0.0, max_tokens: int = 256,
             **extra) -> Dict:
        """Return {"text", "cache_key", "cached", "latency_s", "usage"}."""
        params = dict(temperature=temperature, max_tokens=max_tokens, **extra)
        key = cache_key(self.model, messages, params)
        row = self.db.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
        if row:
            out = json.loads(row[0])
            out.update(cached=True, cache_key=key, latency_s=0.0)
            self._log(out)
            return out
        if not self.base_url:
            raise RuntimeError("LLM_BASE_URL is not set (add it to .env)")
        t0 = time.time()
        data = self._post({"model": self.model, "messages": messages, **params})
        out = {"text": data["choices"][0]["message"]["content"], "usage": data.get("usage", {})}
        self.db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)", (key, json.dumps(out), time.time()))
        self.db.commit()
        out.update(cached=False, cache_key=key, latency_s=round(time.time() - t0, 3))
        self._log(out)
        return out

    def _post(self, payload: Dict) -> Dict:
        headers = {"Authorization": "Bearer %s" % self.api_key} if self.api_key else {}
        last = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(self.base_url + "/chat/completions", json=payload,
                                  headers=headers, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                last = "HTTP %d: %s" % (r.status_code, r.text[:200])
                if r.status_code in (400, 401, 403, 404):
                    break  # retrying will not help
            except requests.RequestException as e:
                last = repr(e)
            if attempt < self.max_retries - 1:
                time.sleep(self.backoff * (2 ** attempt))
        raise RuntimeError("LLM call failed after retries: %s" % last)

    def _log(self, out: Dict) -> None:
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"time": time.time(), "model": self.model, "cache_key": out["cache_key"],
                                "cached": out["cached"], "latency_s": out["latency_s"],
                                "usage": out.get("usage", {})}) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ping", action="store_true", help="send one short request")
    args = p.parse_args(argv)
    client = LLMClient()
    print("LLM_BASE_URL =", client.base_url or "(not set)", "| model =", client.model)
    if args.ping:
        out = client.chat([{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=5)
        print("reply:", out["text"].strip(), "| cached:", out["cached"], "| latency_s:", out["latency_s"])


if __name__ == "__main__":
    main()
