"""Deterministic lexical reranking and duplicate removal for vector candidates."""
from collections import Counter
import math
import re

STOP_WORDS = set("a an and are as at be by do does for from how in is it of on or that the to was what when where which who why with".split())
RERANK_WEIGHT = 0.05


def terms(text: str) -> list[str]:
    return [word for word in re.findall(r"[^\W_]+", text.casefold()) if word not in STOP_WORDS]


def bm25_scores(query: str, texts: list[str]) -> list[float]:
    """BM25 over this candidate pool (k1=1.5, b=0.75), without another model."""
    tokens = [terms(text) for text in texts]
    if not tokens:
        return []
    average = sum(map(len, tokens)) / len(tokens) or 1
    frequencies = Counter(token for document in tokens for token in set(document))
    scores = []
    for document in tokens:
        counts = Counter(document)
        score = 0.0
        for token in set(terms(query)):
            frequency = counts[token]
            if frequency:
                idf = math.log(1 + (len(tokens) - frequencies[token] + 0.5) / (frequencies[token] + 0.5))
                score += idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * len(document) / average))
        scores.append(score)
    return scores


def shingles(text: str) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[^\W_]+", text.casefold())
    width = min(5, len(tokens))
    return {tuple(tokens[i:i + width]) for i in range(len(tokens) - width + 1)} if width else set()


def rank_candidates(
    candidates: list[dict], query: str, *, top_k: int, min_similarity: float | None = None,
    deduplicate: bool = True, rerank: bool = True,
    rerank_weight: float = RERANK_WEIGHT,
) -> list[dict]:
    if not 1 <= top_k <= 1000:
        raise ValueError("top_k must be between 1 and 1000.")
    if min_similarity is not None and (not math.isfinite(min_similarity) or not -1 <= min_similarity <= 1):
        raise ValueError("min_similarity must be a finite number between -1 and 1.")
    if not math.isfinite(rerank_weight) or not 0 <= rerank_weight <= 1:
        raise ValueError("rerank_weight must be a finite number between 0 and 1.")
    selected = [dict(row) for row in candidates if min_similarity is None or row["similarity"] >= min_similarity]
    lexical = bm25_scores(query, [row["text"] for row in selected]) if rerank else [0.] * len(selected)
    maximum = max(lexical, default=0.)
    for row, score in zip(selected, lexical, strict=True):
        row["rerank_score"] = (1 - rerank_weight) * row["similarity"] + rerank_weight * score / maximum if maximum else row["similarity"]
    selected.sort(key=lambda row: (-row["rerank_score"], -row["similarity"], row["source"], row["chunk_index"]))
    result, seen = [], []
    for row in selected:
        fingerprint = shingles(row["text"])
        if deduplicate and any(
            fingerprint == other or (fingerprint and len(fingerprint & other) / len(fingerprint | other) >= 0.8)
            for other in seen
        ):
            continue
        result.append(row)
        seen.append(fingerprint)
        if len(result) >= top_k:
            break
    return result
