from __future__ import annotations

import math
import re
from typing import Any, Iterable, Sequence

_TERM_RE = re.compile(r"[0-9A-Za-z_\uac00-\ud7a3]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def terms(text: str, *, limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for term in _TERM_RE.findall(str(text or "").lower()):
        if len(term) < 2 or term in seen:
            continue
        seen.add(term)
        out.append(term)
        if limit and len(out) >= limit:
            break
    return out


def lexical_similarity(left: str, right: str) -> float:
    a, b = set(terms(left)), set(terms(right))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _rough_tokens(text: str) -> int:
    korean_chars = len(re.findall(r"[\uac00-\ud7a3]", text))
    ascii_words = len(re.findall(r"[0-9A-Za-z_]+", text))
    other_chars = max(0, len(text) - korean_chars)
    return max(1, int(korean_chars / 2.2) + ascii_words + int(other_chars / 5.0))


def _split_sentences(text: str) -> list[str]:
    stripped = re.sub(r"\s+", " ", text).strip()
    if not stripped:
        return []
    parts = [part.strip() for part in _SENTENCE_RE.split(stripped) if part.strip()]
    if len(parts) <= 1 and len(stripped) > 500:
        parts = [stripped[i:i + 350].strip() for i in range(0, len(stripped), 350)]
    return parts or [stripped]


def _quality(chunk: str, *, min_chars: int, target_chars: int, max_chars: int, boundary: str) -> dict[str, float | str | int]:
    size = len(chunk)
    if min_chars <= size <= max_chars:
        size_score = 1.0
    else:
        distance = min(abs(size - min_chars), abs(size - max_chars), abs(size - target_chars))
        size_score = max(0.0, 1.0 - distance / max(1, target_chars))
    return {
        "chars": size,
        "rough_tokens": _rough_tokens(chunk),
        "size_compliance": round(size_score, 4),
        "boundary": boundary,
    }


def adaptive_chunks(text: str, *, target_chars: int = 900, max_chars: int = 1400, min_chars: int = 160) -> list[dict[str, Any]]:
    text = str(text or "").strip()
    if not text:
        return []
    target_chars = max(120, int(target_chars))
    max_chars = max(target_chars, int(max_chars))
    min_chars = max(20, min(int(min_chars), target_chars))

    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if not blocks:
        blocks = [text]

    chunks: list[tuple[str, str]] = []
    current: list[str] = []
    current_len = 0

    def flush(boundary: str = "paragraph") -> None:
        nonlocal current, current_len
        if current:
            chunks.append(("\n\n".join(current).strip(), boundary))
            current = []
            current_len = 0

    for block in blocks:
        if len(block) > max_chars:
            flush()
            sentence_buf: list[str] = []
            sentence_len = 0
            for sentence in _split_sentences(block):
                pieces = [sentence]
                if len(sentence) > max_chars:
                    pieces = [sentence[i:i + max_chars].strip() for i in range(0, len(sentence), max_chars)]
                for piece in pieces:
                    if not piece:
                        continue
                    add_len = len(piece) + (1 if sentence_buf else 0)
                    if sentence_buf and sentence_len + add_len > max_chars:
                        chunks.append((" ".join(sentence_buf).strip(), "sentence"))
                        sentence_buf = []
                        sentence_len = 0
                    sentence_buf.append(piece)
                    sentence_len += add_len
            if sentence_buf:
                chunks.append((" ".join(sentence_buf).strip(), "sentence"))
            continue
        next_len = current_len + len(block) + (2 if current else 0)
        if current and next_len > max_chars:
            flush()
        current.append(block)
        current_len += len(block) + (2 if len(current) > 1 else 0)
        if current_len >= target_chars:
            flush()
    flush()

    merged: list[tuple[str, str]] = []
    for chunk, boundary in chunks:
        if merged and len(chunk) < min_chars and len(merged[-1][0]) + len(chunk) + 2 <= max_chars:
            prev, prev_boundary = merged[-1]
            merged[-1] = (f"{prev}\n\n{chunk}".strip(), prev_boundary)
        else:
            merged.append((chunk, boundary))

    return [
        {
            "index": index,
            "text": chunk,
            "quality": _quality(chunk, min_chars=min_chars, target_chars=target_chars, max_chars=max_chars, boundary=boundary),
        }
        for index, (chunk, boundary) in enumerate(merged)
        if chunk
    ]


def best_snippet(text: str, query: str, *, max_chars: int = 700) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    chunks = adaptive_chunks(text, target_chars=max(220, max_chars // 2), max_chars=max_chars, min_chars=120)
    if not chunks:
        return text[:max_chars].rstrip()
    ranked = sorted(chunks, key=lambda c: lexical_similarity(query, str(c["text"])), reverse=True)
    snippet = str(ranked[0]["text"]).strip()
    return snippet if len(snippet) <= max_chars else snippet[:max_chars].rstrip()


def reciprocal_rank_fusion(candidate_groups: Sequence[Sequence[dict[str, Any]]], *, id_key: str = "id", k: int = 60, weights: Sequence[float] | None = None) -> dict[int, float]:
    scores: dict[int, float] = {}
    if weights is None:
        weights = [1.0] * len(candidate_groups)
    for group_index, group in enumerate(candidate_groups):
        weight = float(weights[group_index]) if group_index < len(weights) else 1.0
        for rank, item in enumerate(group, start=1):
            if id_key not in item:
                continue
            item_id = int(item[id_key])
            scores[item_id] = scores.get(item_id, 0.0) + weight / float(k + rank)
    return scores


def _candidate_text(item: dict[str, Any], fields: Iterable[str]) -> str:
    return " ".join(str(item.get(field) or "") for field in fields)


def mmr_rerank(candidates: Sequence[dict[str, Any]], *, query: str, limit: int, relevance_key: str = "score", text_fields: Sequence[str] = ("title", "content"), diversity: float = 0.28) -> list[dict[str, Any]]:
    remaining = [dict(item) for item in candidates]
    picked: list[dict[str, Any]] = []
    if not remaining or limit <= 0:
        return []
    diversity = max(0.0, min(0.9, float(diversity)))
    relevance_weight = 1.0 - diversity
    max_relevance = max(float(item.get(relevance_key) or 0.0) for item in remaining) or 1.0
    while remaining and len(picked) < limit:
        best_index = 0
        best_score = -math.inf
        for index, item in enumerate(remaining):
            relevance = float(item.get(relevance_key) or 0.0) / max_relevance
            novelty_penalty = 0.0
            if picked:
                text = _candidate_text(item, text_fields)
                novelty_penalty = max(lexical_similarity(text, _candidate_text(other, text_fields)) for other in picked)
            mmr_score = relevance_weight * relevance - diversity * novelty_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_index = index
        selected = remaining.pop(best_index)
        selected["mmr_score"] = round(best_score, 4)
        picked.append(selected)
    return picked


def build_context_pack(query: str, candidates: Sequence[dict[str, Any]], *, max_chars: int = 3200, per_item_chars: int = 700, limit: int = 8) -> dict[str, Any]:
    selected = mmr_rerank(candidates, query=query, limit=limit, diversity=0.30)
    budget = max(400, int(max_chars))
    used = 0
    items: list[dict[str, Any]] = []
    for item in selected:
        title = str(item.get("title") or "").strip()
        snippet = best_snippet(str(item.get("content") or item.get("summary") or ""), query, max_chars=per_item_chars)
        record = {
            "id": item.get("id"),
            "title": title,
            "type": item.get("memory_type") or item.get("type") or item.get("node_type"),
            "score": item.get("score") or item.get("activation_score"),
            "snippet": snippet,
        }
        record_len = len(title) + len(snippet)
        if items and used + record_len > budget:
            break
        items.append(record)
        used += record_len
    return {
        "query": query,
        "max_chars": budget,
        "used_chars": used,
        "candidate_count": len(candidates),
        "item_count": len(items),
        "items": items,
    }
