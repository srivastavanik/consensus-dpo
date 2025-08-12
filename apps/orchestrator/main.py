from __future__ import annotations

import os
from typing import List, Optional

import orjson
from fastapi import FastAPI
from pydantic import BaseModel

from libs.consensus_dpo.provider import CompletionRequest, GenParams, NovitaClient
from libs.consensus_dpo.datasets import PairBuilder


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


@app.post("/consensus")
async def consensus(req: ConsensusRequest) -> dict:
    client = NovitaClient()
    gen_params = GenParams(temperature=0.9, top_p=0.95, max_tokens=512)
    reqs = [CompletionRequest(model=req.model, prompt=req.prompt, params=gen_params) for _ in range(req.k)]
    cands = await client.batchGenerate(reqs)

    # Minimal judge: compare first two with a simple prompt using the judge worker template
    judge_prompt = (
        "You are a careful judge. Compare two answers (A,B) for the same task.\n"
        "Apply bias controls: ignore style; equalize length; consider evidence.\n"
        "Return JSON: {winner:'A|B|Tie', reasons:['...','...'], score_delta:-3..3, pos_swap_consistency:true|false, len_norm_consistency:true|false}.\n"
        f"Task: {req.prompt}\nA: {cands[0].text}\nB: {cands[1].text}\n"
    )
    judge_out = await client.generate(CompletionRequest(model=req.model, prompt=judge_prompt, params=GenParams(temperature=0.2, top_p=0.9, max_tokens=200)))

    # Try to parse as JSON; if fails, fallback to a Tie
    import json as _json

    try:
        decision = _json.loads(judge_out.text)
    except Exception:
        decision = {"winner": "Tie", "score_delta": 0, "pos_swap_consistency": False, "len_norm_consistency": False}

    from libs.consensus_dpo.datasets.pairs import Candidate

    cand_a = Candidate(answer=cands[0].text, rationale="", citations=[])
    cand_b = Candidate(answer=cands[1].text, rationale="", citations=[])
    out_path = os.getenv("PAIRS_OUT", "./data/pairs.v1.jsonl")
    builder = PairBuilder(out_path)
    rec = builder.add_pair(req.prompt, cand_a, cand_b, decision)
    await client.aclose()
    return {"decision": decision, "pair_written": bool(rec), "pairs_path": out_path}


