from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Candidate:
    answer: str
    rationale: str = ""
    citations: List[str] = None  # type: ignore


@dataclass
class JudgeMeta:
    score_delta: int
    pos_consistent: bool
    len_consistent: bool


@dataclass
class PairRecord:
    prompt: str
    chosen: Candidate
    rejected: Candidate
    judge: JudgeMeta
    debate_meta: Dict[str, Any]
    tools: Dict[str, Any]


class PairBuilder:
    """Build JSONL DPO pairs with filtering and provenance logging."""

    def __init__(self, out_path: str) -> None:
        self.out_path = out_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def _passes_filters(self, judge: JudgeMeta, a_len: int, b_len: int) -> bool:
        if not (judge.pos_consistent and judge.len_consistent):
            return False
        # Penalize verbosity wins (>1.5× length)
        longer = max(a_len, b_len)
        shorter = min(a_len, b_len) or 1
        if longer / shorter > 1.5:
            return False
        return True

    def add_pair(
        self,
        prompt: str,
        cand_a: Candidate,
        cand_b: Candidate,
        judge_decision: Dict[str, Any],
        debate_meta: Optional[Dict[str, Any]] = None,
        tools: Optional[Dict[str, Any]] = None,
    ) -> Optional[PairRecord]:
        winner = (judge_decision.get("winner") or "").upper()
        if winner not in {"A", "B"}:
            return None
        meta = JudgeMeta(
            score_delta=int(judge_decision.get("score_delta", 0)),
            pos_consistent=bool(judge_decision.get("pos_swap_consistency", False)),
            len_consistent=bool(judge_decision.get("len_norm_consistency", False)),
        )

        if not self._passes_filters(meta, len(cand_a.answer), len(cand_b.answer)):
            return None

        if winner == "A":
            chosen, rejected = cand_a, cand_b
        else:
            chosen, rejected = cand_b, cand_a

        rec = PairRecord(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            judge=meta,
            debate_meta=debate_meta or {"rounds": 0, "agents": 0},
            tools=tools or {},
        )
        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        return rec


