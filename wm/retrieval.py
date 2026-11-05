"""Small BM25 retriever (pure Python) for the 3-shot sanity check."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [Counter(tokenize(d)) for d in docs]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> List[float]:
        q = tokenize(query)
        out = []
        for d, length in zip(self.docs, self.lengths):
            s = 0.0
            for t in q:
                f = d.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg))
            out.append(s)
        return out

    def top_k(self, query: str, k: int) -> List[int]:
        s = self.scores(query)
        return sorted(range(len(s)), key=lambda i: (-s[i], i))[:k]
