"""Frozen-Qwen RAG arm and model-free retrieval-relevance metrics."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
RETRIEVER_MODEL = "nvidia/NV-Embed-v2"  # the encoder JEPA (arm D) uses, 4-bit
K = 5
STOPWORDS = r"\b(the|a|an|to|at|in|on|with|from)\b"
# Characters kept per text in the prompt: 5 examples x 2 texts x 1500 + a 4000-char state stay
# under ~5k tokens, well inside vLLM's 8192 (most texts are far shorter and are not cut).
DEMO_CHARS, QUERY_CHARS = 1500, 4000


def first(row: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    return default


def action_template(action: str) -> str:
    """Verb plus bracketed command shape; robust to ALFWorld/ScienceWorld text."""
    action = action.lower().strip()
    action = re.sub(STOPWORDS, "", action)
    return re.sub(r"\b[a-z0-9_-]+\b", lambda m: "<x>" if m.start() > 0 else m.group(), action, count=0)


def action_object(action: str) -> str:
    """First word after the verb, ignoring stopwords ("go to drawer 4" -> "drawer", not "to")."""
    words = re.findall(r"[a-z0-9_-]+", re.sub(STOPWORDS, "", action.lower()))
    return words[1] if len(words) > 1 else ""


def relevant(query_action: str, candidate_action: str) -> bool:
    """Model-free relevance: same action template AND primary object."""
    return (action_template(query_action) == action_template(candidate_action)
            and action_object(query_action) == action_object(candidate_action))


@dataclass(frozen=True)
class Transition:
    observation: str
    action: str
    next_observation: str
    raw: dict[str, Any]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def transition(row: dict[str, Any]) -> Transition:
    return Transition(
        first(row, "observation", "obs", "state", "current_observation"),
        first(row, "action", "executed_action", "gold_action"),
        first(row, "next_observation", "obs_next", "next_obs", "next_state", "target"), row,
    )


class NVEmbedEncoder:
    """NV-Embed-v2 through the team's wrapper (jepa/encoder/text_encoder.py): 4-bit on GPU,
    L2-normalised, batched. Load it once per process: it needs most of the 12 GB GPU."""
    def __init__(self, device: str = "cuda"):
        from jepa.encoder.text_encoder import TextEncoder
        self.encoder = TextEncoder(model_name=RETRIEVER_MODEL, quantize=(device != "cpu"), device=device)

    def encode(self, texts, normalize_embeddings=True, batch_size=64):
        return self.encoder.encode(list(texts)).numpy()


class CachedEncoder:
    """Looks texts up in vectors computed once (prepare_g_prompts.py encodes every text of a set
    in one pass, then builds the retriever for each budget from this cache)."""
    def __init__(self, vectors: dict[str, np.ndarray]):
        self.vectors = vectors

    def encode(self, texts, normalize_embeddings=True, batch_size=64):
        return np.stack([self.vectors[t] for t in texts])


class HierarchicalRetriever:
    """Top-min(50,n/2) observation retrieval, action-similarity re-rank, top-k."""
    def __init__(self, buffer: Iterable[dict[str, Any]], encoder_name: str = RETRIEVER_MODEL,
                 device: str | None = None, encoder: Any = None):
        self.items = [transition(x) for x in buffer]
        if not self.items:
            raise ValueError("Retrieval buffer is empty")
        if encoder is None:
            if encoder_name != RETRIEVER_MODEL:
                raise ValueError(f"only {RETRIEVER_MODEL} is wired up")
            encoder = NVEmbedEncoder(device or "cuda")
        self.encoder = encoder
        self.obs = self.encode([x.observation for x in self.items])
        self.actions = self.encode([x.action for x in self.items])

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.encoder.encode(texts, normalize_embeddings=True, batch_size=64))

    def retrieve(self, observation: str, candidate_action: str, k: int = K) -> list[Transition]:
        q_obs, q_action = self.encode([observation, candidate_action])
        return self.retrieve_vec(q_obs, q_action, k)

    def retrieve_vec(self, q_obs: np.ndarray, q_action: np.ndarray, k: int = K) -> list[Transition]:
        """retrieve() from already-encoded query vectors (the generator encodes in batches)."""
        pre_k = min(50, max(1, len(self.items) // 2))
        obs_order = np.argsort(-(self.obs @ q_obs))[:pre_k]
        ranked = obs_order[np.argsort(-(self.actions[obs_order] @ q_action))]
        return [self.items[int(i)] for i in ranked[:k]]


def recall_at5(rows: Iterable[dict[str, Any]], retriever: HierarchicalRetriever) -> float:
    rows = list(rows)
    if not rows:
        return float("nan")
    hits = 0
    for row in rows:
        t = transition(row)
        retrieved = retriever.retrieve(t.observation, t.action)
        hits += any(relevant(t.action, x.action) for x in retrieved)
    return hits / len(rows)


def clip(text: str, limit: int = DEMO_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + " ..."


def prompt(observation: str, action: str, examples: list[Transition]) -> str:
    demos = "\n\n".join(
        f"Observation: {clip(x.observation)}\nAction: {x.action}\nNext observation: {clip(x.next_observation)}"
        for x in examples
    )
    return ("Predict the next text observation after the action. Return only the prediction.\n\n"
            f"Retrieved transitions:\n{demos}\n\n"
            f"Observation: {clip(observation, QUERY_CHARS)}\nAction: {action}\nNext observation:")


class FrozenQwen:
    def __init__(self, model_name: str = QWEN_MODEL):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")

    def predict(self, text: str) -> str:
        messages = [{"role": "user", "content": text}]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([rendered], return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        return self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


class VLLMQwen:
    """The same frozen Qwen through the team's vLLM server (wm/llm_client.py: cached, retried).

    The 12 GB RTX 5070 cannot hold full-precision 7B weights, so the server runs
    Qwen2.5-7B-Instruct-AWQ (the judge's model). Greedy, 128 new tokens, as in FrozenQwen.
    """
    def __init__(self):
        from wm.llm_client import LLMClient
        self.client = LLMClient()

    def predict(self, text: str) -> str:
        out = self.client.chat([{"role": "user", "content": text}], temperature=0.0, max_tokens=128)
        return out["text"].strip()
