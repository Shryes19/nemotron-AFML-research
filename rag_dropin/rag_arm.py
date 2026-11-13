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


def first(row: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    return default


def action_template(action: str) -> str:
    """Verb plus bracketed command shape; robust to ALFWorld/ScienceWorld text."""
    action = action.lower().strip()
    action = re.sub(r"\b(the|a|an|to|at|in|on|with|from)\b", "", action)
    return re.sub(r"\b[a-z0-9_-]+\b", lambda m: "<x>" if m.start() > 0 else m.group(), action, count=0)


def action_object(action: str) -> str:
    words = re.findall(r"[a-z0-9_-]+", action.lower())
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
        first(row, "next_observation", "next_obs", "next_state", "target"), row,
    )


class HierarchicalRetriever:
    """Top-min(50,n/2) observation retrieval, action-similarity re-rank, top-k."""
    def __init__(self, buffer: Iterable[dict[str, Any]], encoder_name: str = RETRIEVER_MODEL):
        from sentence_transformers import SentenceTransformer
        self.items = [transition(x) for x in buffer]
        if not self.items:
            raise ValueError("Retrieval buffer is empty")
        self.encoder = SentenceTransformer(encoder_name)
        self.obs = self.encoder.encode([x.observation for x in self.items], normalize_embeddings=True)
        self.actions = self.encoder.encode([x.action for x in self.items], normalize_embeddings=True)

    def retrieve(self, observation: str, candidate_action: str, k: int = K) -> list[Transition]:
        q_obs = self.encoder.encode([observation], normalize_embeddings=True)[0]
        q_action = self.encoder.encode([candidate_action], normalize_embeddings=True)[0]
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


def prompt(observation: str, action: str, examples: list[Transition]) -> str:
    demos = "\n\n".join(
        f"Observation: {x.observation}\nAction: {x.action}\nNext observation: {x.next_observation}"
        for x in examples
    )
    return ("Predict the next text observation after the action. Return only the prediction.\n\n"
            f"Retrieved transitions:\n{demos}\n\nObservation: {observation}\nAction: {action}\nNext observation:")


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
