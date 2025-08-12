from __future__ import annotations

from typing import Dict


def pick_chosen_text(final_decision: Dict, a_text: str, b_text: str) -> str:
    winner = (final_decision or {}).get("winner", "Tie")
    if winner == "A":
        return a_text
    if winner == "B":
        return b_text
    # Tie: fallback to shorter answer
    return a_text if len(a_text) <= len(b_text) else b_text


