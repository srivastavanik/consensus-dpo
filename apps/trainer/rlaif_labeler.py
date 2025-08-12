from __future__ import annotations

import json
import os
from typing import Dict, List

from libs.consensus_dpo.provider import CompletionRequest, GenParams, NovitaClient
from libs.consensus_dpo.utils.json_utils import extract_json_object
from libs.consensus_dpo.datasets.pairs import Candidate, PairBuilder
from libs.consensus_dpo.prompts import JUDGE_TEMPLATE


async def label_with_teacher(prompts: List[str], student_model: str, teacher_model: str, out_path: str) -> None:
    client = NovitaClient()
    builder = PairBuilder(out_path)
    for p in prompts:
        # get two diverse student answers via higher temperature
        gen_params = GenParams(temperature=0.95, top_p=0.95, max_tokens=512)
        reqs = [CompletionRequest(model=student_model, prompt=p, params=gen_params) for _ in range(2)]
        outs = await client.batchGenerate(reqs)
        a_text, b_text = outs[0].text, outs[1].text
        # judge via teacher model
        jprompt = JUDGE_TEMPLATE.format(problem=p, a=a_text, b=b_text)
        judge = await client.generate(CompletionRequest(model=teacher_model, prompt=jprompt, params=GenParams(temperature=0.1, top_p=0.9, max_tokens=220)))
        decision = extract_json_object(judge.text) or {"winner": "Tie", "score_delta": 0, "pos_swap_consistency": False, "len_norm_consistency": False}
        builder.add_pair(p, Candidate(answer=a_text, rationale="", citations=[]), Candidate(answer=b_text, rationale="", citations=[]), decision, debate_meta={"rounds": 0, "agents": 2})
    await client.aclose()


