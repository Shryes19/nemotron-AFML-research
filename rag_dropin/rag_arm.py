"""Frozen-Qwen RAG arm and model-free retrieval-relevance metrics."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
RETRIEVER_MODEL = "BAAI/bge-base-en-v1.5"  # Not NV-Embed.
K = 5
STOPWORDS = r"\b(the|a|an|to|at|in|on|with|from)\b"
MAX_CHARS = 3000  # per text in the prompt; keeps 5 ScienceWorld examples well inside 8192 tokens


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


class HierarchicalRetriever:
    """Top-min(50,n/2) observation retrieval, action-similarity re-rank, top-k."""
    def __init__(self, buffer: Iterable[dict[str, Any]], encoder_name: str = RETRIEVER_MODEL,
                 device: str | None = None, encoder: Any = None):
        self.items = [transition(x) for x in buffer]
        if not self.items:
            raise ValueError("Retrieval buffer is empty")
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(encoder_name, device=device)
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


def clip(text: str) -> str:
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + " ..."


def prompt(observation: str, action: str, examples: list[Transition]) -> str:
    demos = "\n\n".join(
        f"Observation: {clip(x.observation)}\nAction: {x.action}\nNext observation: {clip(x.next_observation)}"
        for x in examples
    )
    return ("Predict the next text observation after the action. Return only the prediction.\n\n"
            f"Retrieved transitions:\n{demos}\n\nObservation: {clip(observation)}\nAction: {action}\nNext observation:")


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
