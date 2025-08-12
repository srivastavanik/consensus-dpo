from __future__ import annotations

import asyncio
import json
import os
from typing import Dict

import orjson
import redis

from libs.consensus_dpo.provider import CompletionRequest, GenParams, NovitaClient


IN_Q = os.getenv("JUDGE_QUEUE_IN", "queue:judge:in")
OUT_Q = os.getenv("JUDGE_QUEUE_OUT", "queue:judge:out")

JUDGE_TEMPLATE = (
    "You are a careful judge. Compare two answers (A,B) for the same task.\n"
    "Apply bias controls: ignore style; equalize length; consider evidence.\n"
    "Return JSON: {{winner:'A|B|Tie', reasons:['...','...'], score_delta:-3..3, pos_swap_consistency:true|false, len_norm_consistency:true|false}}.\n"
    "Task: {problem}\nA: {a}\nB: {b}\n"
)


async def judge_loop() -> None:
    r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    client = NovitaClient()
    try:
        while True:
            _, payload = r.blpop(IN_Q)
            task: Dict = json.loads(payload)
            prompt = JUDGE_TEMPLATE.format(problem=task["problem"], a=task["a"], b=task["b"])
            params = GenParams(temperature=0.2, top_p=0.9, max_tokens=200)
            req = CompletionRequest(model=task["model"], prompt=prompt, params=params)
            out = await client.generate(req)
            r.rpush(OUT_Q, orjson.dumps({"id": task.get("id"), "decision": out.text}))
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(judge_loop())


