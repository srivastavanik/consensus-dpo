from __future__ import annotations

GENERATOR_TEMPLATE = (
    "You are a careful, concise reasoner.\n"
    "Task: {problem}\n"
    "Rules: Show steps succinctly; cite facts with [DocID] or URL; do not fabricate.\n"
    'Return JSON: {"answer": "...", "rationale": "...", "citations": ["..."]}'
)

DEBATE_R1_TEMPLATE = (
    "Cross-examination. Each candidate receives a peer’s answer; point to specific mistakes or add independent checks (<=120 tokens).\n"
    "Task: {problem}\nPeer: {peer}\n"
)

DEBATE_R2_TEMPLATE = (
    "Defense/Revision. Revise or defend and set changed: true|false. Final brief 80–120 tokens with explicit evidence references.\n"
    "Task: {problem}\nYou (prev): {self_prev}\nCritique: {critique}\n"
)

JUDGE_TEMPLATE = (
    "Bias-controlled LLM-as-judge. Swap A/B ordering, equalize lengths, blind model IDs, normalize style markers.\n"
    "Return JSON: { winner: 'A|B|Tie', reasons: ['...','...'], score_delta: -3..3, pos_swap_consistency: true|false, len_norm_consistency: true|false }\n"
    "Task: {problem}\nA: {a}\nB: {b}"
)


