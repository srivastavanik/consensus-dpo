from __future__ import annotations

import re
from typing import Dict, List, Tuple


def simple_string_match_support(answer: str, span_text: str) -> float:
    """Heuristic: proportion of unique content words from answer found in span."""
    tok = re.findall(r"[A-Za-z0-9]+", answer.lower())
    span = set(re.findall(r"[A-Za-z0-9]+", span_text.lower()))
    if not tok:
        return 0.0
    content = [t for t in tok if len(t) > 3]
    if not content:
        return 0.0
    hit = sum(1 for t in set(content) if t in span)
    return hit / len(set(content))


def verify_citations(answer: str, cited_docs: List[Dict]) -> Tuple[bool, List[Dict]]:
    """Return (is_supported, scores_per_doc)."""
    scores = []
    for d in cited_docs:
        score = simple_string_match_support(answer, d.get("text", ""))
        scores.append({"doc_id": d.get("doc_id"), "score": score})
    is_supported = any(s["score"] >= 0.3 for s in scores)
    return is_supported, scores


