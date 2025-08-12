from __future__ import annotations

import os
from typing import List, Optional

import orjson
from fastapi import FastAPI
from pydantic import BaseModel

from libs.consensus_dpo.provider import CompletionRequest, GenParams, NovitaClient


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


