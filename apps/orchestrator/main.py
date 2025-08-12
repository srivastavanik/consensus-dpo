from __future__ import annotations

import os
from typing import List, Optional

import orjson
from fastapi import FastAPI
from pydantic import BaseModel

from libs.consensus_dpo.provider import CompletionRequest, GenParams, NovitaClient
from libs.consensus_dpo.datasets import PairBuilder
from libs.consensus_dpo.prompts import GENERATOR_TEMPLATE, JUDGE_TEMPLATE
from libs.consensus_dpo.utils.json_utils import extract_json_object


class GenerateRequest(BaseModel):
    prompt: str
    model: str
    k: int = 3
    temperature: float = 0.8
    top_p: float = 0.9
    max_tokens: int = 512


class GenerateResponse(BaseModel):
    candidates: List[str]
    usage: Optional[dict] = None


app = FastAPI(title="Consensus-DPO Orchestrator")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    client = NovitaClient()
    params = GenParams(
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
    )
    tasks = [
        CompletionRequest(model=req.model, prompt=req.prompt, params=params)
        for _ in range(req.k)
    ]
    outs = await client.batchGenerate(tasks)
    await client.aclose()
    return GenerateResponse(candidates=[o.text for o in outs], usage=outs[0].usage if outs else {})


class ConsensusRequest(BaseModel):
    prompt: str
    model: str
    k: int = 3
    m: int = 2  # counterfactual judge views
    r: int = 1  # debate rounds (R=1 minimal now)


@app.post("/consensus")
async def consensus(req: ConsensusRequest) -> dict:
    client = NovitaClient()
    # 1) Generation with structured prompt
    gen_params = GenParams(temperature=0.9, top_p=0.95, max_tokens=512)
    prompts = [
        GENERATOR_TEMPLATE.format(problem=req.prompt)
        for _ in range(req.k)
    ]
    gen_reqs = [CompletionRequest(model=req.model, prompt=p, params=gen_params) for p in prompts]
    cands = await client.batchGenerate(gen_reqs)

    # 2) Debate R=1 minimal (pairwise cross-exam skipped for brevity; next commit will add)

    # 3) Judge with m counterfactual views (swap A/B)
    a_text = cands[0].text
    b_text = cands[1].text if len(cands) > 1 else cands[0].text

    decisions = []
    judge_params = GenParams(temperature=0.2, top_p=0.9, max_tokens=220)
    # View 1: A,B
    judge_p1 = JUDGE_TEMPLATE.format(problem=req.prompt, a=a_text, b=b_text)
    # View 2: B,A
    judge_p2 = JUDGE_TEMPLATE.format(problem=req.prompt, a=b_text, b=a_text)
    for p in [judge_p1, judge_p2][: max(1, req.m)]:
        out = await client.generate(CompletionRequest(model=req.model, prompt=p, params=judge_params))
        parsed = extract_json_object(out.text) or {
            "winner": "Tie",
            "score_delta": 0,
            "pos_swap_consistency": False,
            "len_norm_consistency": False,
        }
        decisions.append(parsed)

    # Aggregate: consistency if both views flip winner accordingly
    if len(decisions) >= 2:
        w1, w2 = decisions[0].get("winner"), decisions[1].get("winner")
        # If we swapped inputs in view2, then consistency means w2 is the opposite
        pos_consistent = ((w1 == "A" and w2 == "B") or (w1 == "B" and w2 == "A") or (w1 == "Tie" and w2 == "Tie"))
        for d in decisions:
            d["pos_swap_consistency"] = bool(pos_consistent)

    final_decision = decisions[0]

    from libs.consensus_dpo.datasets.pairs import Candidate

    cand_a = Candidate(answer=a_text, rationale="", citations=[])
    cand_b = Candidate(answer=b_text, rationale="", citations=[])
    out_path = os.getenv("PAIRS_OUT", "./data/pairs.v1.jsonl")
    builder = PairBuilder(out_path)
    rec = builder.add_pair(req.prompt, cand_a, cand_b, final_decision, debate_meta={"rounds": req.r, "agents": req.k})
    await client.aclose()
    return {"decisions": decisions, "final": final_decision, "pair_written": bool(rec), "pairs_path": out_path}


