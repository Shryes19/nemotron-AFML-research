"""LLM client and policy against a fake local OpenAI-style server (no GPU needed):
  python -m pytest tests/test_llm.py -q
"""
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from wm.llm_client import LLMClient, cache_key
from wm.llm_policy import LLMPolicy, candidates, parse_choice
from wm.obs import build_obs


class FakeServer:
    """Replies with self.reply; the first self.fail_first requests get HTTP 500."""

    def __init__(self):
        self.calls, self.fail_first, self.reply = 0, 0, "2"
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.calls += 1
                owner.last_body, owner.last_auth = body, self.headers.get("Authorization")
                if owner.calls <= owner.fail_first:
                    self.send_response(500)
                    self.end_headers()
                    return
                data = {"choices": [{"message": {"content": owner.reply}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d/v1" % self.httpd.server_port
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.httpd.shutdown()


@pytest.fixture
def client(server, tmp_path):
    return LLMClient(base_url=server.url, model="test-model", api_key="k", backoff=0.01,
                     cache_path=str(tmp_path / "c.sqlite"), log_path=str(tmp_path / "log.jsonl"))


MSG = [{"role": "user", "content": "hi"}]


def test_cache_hit_avoids_second_call(client, server):
    a = client.chat(MSG)
    b = client.chat(MSG)
    assert server.calls == 1 and not a["cached"] and b["cached"]
    assert a["text"] == b["text"] == "2"
    assert a["cache_key"] == cache_key("test-model", MSG, {"temperature": 0.0, "max_tokens": 256})
    assert server.last_auth == "Bearer k" and server.last_body["model"] == "test-model"


def test_different_prompt_is_a_miss(client, server):
    client.chat(MSG)
    client.chat([{"role": "user", "content": "hello"}])
    assert server.calls == 2


def test_retry_then_success(client, server):
    server.fail_first = 2
    assert client.chat(MSG)["text"] == "2"
    assert server.calls == 3


def test_gives_up_after_max_retries(client, server):
    server.fail_first = 100
    with pytest.raises(RuntimeError):
        client.chat(MSG)
    assert server.calls == client.max_retries


def test_parse_choice():
    assert parse_choice("3", 5) == 2
    assert parse_choice("Action 1.", 5) == 0
    assert parse_choice("9", 5) is None and parse_choice("none", 5) is None


def test_candidates_capped_and_deterministic():
    acts = ["connect a%d to b" % i for i in range(500)] + ["open door", "look around"]
    c1, c2 = candidates(acts, random.Random(0)), candidates(acts, random.Random(0))
    assert c1 == c2 and len(c1) == 60 and "open door" in c1 and "look around" in c1


class FakeEnv:
    def admissible_actions(self):
        return ["go to desk 1", "open drawer 1", "help"]


def test_policy_choice_and_fallback(client, server):
    obs = build_obs("goal", "room", "loc", "fb", "inv", ["go to desk 1", "open drawer 1"])
    policy = LLMPolicy("alfworld", client=client)
    action, meta = policy.act(FakeEnv(), random.Random(0), obs_text=obs)
    assert action == "open drawer 1" and "fallback" not in meta
    assert meta["llm_meta"]["model"] == "test-model"
    server.reply = "I am not sure"
    obs2 = build_obs("other goal", "room", "loc", "fb", "inv", [])  # new prompt, so not cached
    action, meta = policy.act(FakeEnv(), random.Random(1), obs_text=obs2)
    assert meta["fallback"] and action in ("go to desk 1", "open drawer 1")
