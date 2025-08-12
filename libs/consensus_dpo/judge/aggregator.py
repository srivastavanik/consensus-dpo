from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class JudgeView:
    winner: str
    score_delta: int
    pos_swap_consistency: bool
    len_norm_consistency: bool


def aggregate_views(views: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate multiple judge views.

    Policy: drop if any inconsistency; else pick first non-tie; average score.
    """
    if not views:
        return {"winner": "Tie", "score_delta": 0, "pos_swap_consistency": False, "len_norm_consistency": False}
    if any(not (v.get("pos_swap_consistency") and v.get("len_norm_consistency")) for v in views):
        return {"winner": "Tie", "score_delta": 0, "pos_swap_consistency": False, "len_norm_consistency": False}
    winners = [v.get("winner") for v in views]
    pick = next((w for w in winners if w in {"A", "B"}), "Tie")
    avg_score = round(sum(int(v.get("score_delta", 0)) for v in views) / max(1, len(views)))
    return {
        "winner": pick,
        "score_delta": int(avg_score),
        "pos_swap_consistency": True,
        "len_norm_consistency": True,
    }


