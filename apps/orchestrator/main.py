from __future__ import annotations

import os
from typing import List, Optional

import orjson
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow

from libs.consensus_dpo.provider import CompletionRequest, GenParams, NovitaClient
from libs.consensus_dpo.datasets import PairBuilder
from libs.consensus_dpo.prompts import GENERATOR_TEMPLATE, JUDGE_TEMPLATE
from libs.consensus_dpo.utils.json_utils import extract_json_object
from libs.consensus_dpo.utils.parse_structured import parse_generator_json
from libs.consensus_dpo.judge.aggregator import aggregate_views
from libs.consensus_dpo.debate.r1r2 import run_r1, run_r2
from libs.consensus_dpo.retrieval.verifier import verify_citations


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
    # Start MLflow run for observability
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "consensus-dpo"))
    mlflow.start_run(run_name="consensus")
    # 1) Generation with structured prompt
    gen_params = GenParams(temperature=0.9, top_p=0.95, max_tokens=512)
    prompts = [
        GENERATOR_TEMPLATE.format(problem=req.prompt)
        for _ in range(req.k)
    ]
    gen_reqs = [CompletionRequest(model=req.model, prompt=p, params=gen_params) for p in prompts]
    cands = await client.batchGenerate(gen_reqs)
    parsed = [parse_generator_json(c.text) or {"answer": c.text, "rationale": "", "citations": []} for c in cands]
    mlflow.log_params({"k": req.k, "m": req.m, "r": req.r})
    mlflow.log_dict({"prompt": req.prompt, "generations": parsed}, artifact_file="generations.json")

    # 2) Debate R=1 minimal: let candidate A critique B and B critique A; both revise
    if req.r >= 1 and len(parsed) >= 2:
        r1_a = await run_r1(req.prompt, parsed[1]["answer"], req.model, client)
        r1_b = await run_r1(req.prompt, parsed[0]["answer"], req.model, client)
        r2_a = await run_r2(req.prompt, parsed[0]["answer"], r1_a.get("critique", ""), req.model, client)
        r2_b = await run_r2(req.prompt, parsed[1]["answer"], r1_b.get("critique", ""), req.model, client)
        # Use revised answers if provided
        if isinstance(r2_a.get("answer"), str) and r2_a.get("answer"):
            parsed[0]["answer"] = r2_a["answer"]
        if isinstance(r2_b.get("answer"), str) and r2_b.get("answer"):
            parsed[1]["answer"] = r2_b["answer"]

    # 3) Judge with m counterfactual views (swap A/B)
    a_text = parsed[0]["answer"]
    b_text = parsed[1]["answer"] if len(parsed) > 1 else parsed[0]["answer"]

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
    mlflow.log_dict({"decisions": decisions}, artifact_file="decisions.json")

    # Aggregate: consistency if both views flip winner accordingly
    if len(decisions) >= 2:
        w1, w2 = decisions[0].get("winner"), decisions[1].get("winner")
        # If we swapped inputs in view2, then consistency means w2 is the opposite
        pos_consistent = ((w1 == "A" and w2 == "B") or (w1 == "B" and w2 == "A") or (w1 == "Tie" and w2 == "Tie"))
        for d in decisions:
            d["pos_swap_consistency"] = bool(pos_consistent)

    final_decision = aggregate_views(decisions)

    from libs.consensus_dpo.datasets.pairs import Candidate

    cand_a = Candidate(answer=a_text, rationale=parsed[0].get("rationale", ""), citations=parsed[0].get("citations", []))
    cand_b = Candidate(answer=b_text, rationale=(parsed[1].get("rationale", "") if len(parsed) > 1 else ""), citations=(parsed[1].get("citations", []) if len(parsed) > 1 else []))

    # Optional: evidence verification if retriever is live
    import httpx as _httpx
    cited_docs = []
    try:
        async with _httpx.AsyncClient(timeout=10) as _hc:
            for c in (cand_a.citations + cand_b.citations):
                resp = await _hc.get(os.getenv("RETRIEVER_URL", "http://127.0.0.1:8010") + "/fetch", params={"doc_id": c})
                if resp.status_code == 200:
                    d = resp.json()
                    d["doc_id"] = c
                    cited_docs.append(d)
    except Exception:
        pass
    supported, support_scores = verify_citations(a_text + "\n" + b_text, cited_docs)
    mlflow.log_dict({"evidence_supported": supported, "support_scores": support_scores}, artifact_file="evidence.json")
    out_path = os.getenv("PAIRS_OUT", "./data/pairs.v1.jsonl")
    builder = PairBuilder(out_path)
    rec = builder.add_pair(req.prompt, cand_a, cand_b, final_decision, debate_meta={"rounds": req.r, "agents": req.k})
    if rec:
        mlflow.log_dict(final_decision, artifact_file="final_decision.json")
    mlflow.end_run()
    await client.aclose()
    return {"decisions": decisions, "final": final_decision, "pair_written": bool(rec), "pairs_path": out_path, "evidence_supported": supported}


