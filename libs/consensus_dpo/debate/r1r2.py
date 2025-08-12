from __future__ import annotations

from typing import Any, Dict, List

from ..prompts import DEBATE_R1_TEMPLATE, DEBATE_R2_TEMPLATE
from ..provider import CompletionRequest, GenParams, NovitaClient
from ..utils.json_utils import extract_json_object


async def run_r1(problem: str, peer_answer: str, model: str, client: NovitaClient) -> Dict[str, Any]:
    prompt = DEBATE_R1_TEMPLATE.format(problem=problem, peer=peer_answer)
    out = await client.generate(
        CompletionRequest(model=model, prompt=prompt, params=GenParams(temperature=0.7, top_p=0.9, max_tokens=160))
    )
    parsed = extract_json_object(out.text) or {"critique": out.text, "checks": []}
    return parsed


async def run_r2(problem: str, self_prev: str, critique: str, model: str, client: NovitaClient) -> Dict[str, Any]:
    prompt = DEBATE_R2_TEMPLATE.format(problem=problem, self_prev=self_prev, critique=critique)
    out = await client.generate(
        CompletionRequest(model=model, prompt=prompt, params=GenParams(temperature=0.7, top_p=0.9, max_tokens=220))
    )
    parsed = extract_json_object(out.text) or {"answer": self_prev, "changed": False, "brief": "", "citations": []}
    return parsed


