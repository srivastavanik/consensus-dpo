from __future__ import annotations

import asyncio
import json
import os
from typing import Dict, List

import orjson
import redis

from libs.consensus_dpo.provider import CompletionRequest, GenParams, NovitaClient


IN_Q = os.getenv("DEBATE_QUEUE_IN", "queue:debate:in")
OUT_Q = os.getenv("DEBATE_QUEUE_OUT", "queue:debate:out")

DEBATE_TEMPLATE = (
    "You will critique and improve an answer.\n"
    "Task: {problem}\n"
    "Peer Answer: {peer}\n"
    "Rules: Be specific; point to errors; add independent checks; <=120 tokens."
)


async def debate_loop() -> None:
    r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    client = NovitaClient()
    try:
        while True:
            _, payload = r.blpop(IN_Q)
            task: Dict = json.loads(payload)
            prompt = DEBATE_TEMPLATE.format(problem=task["problem"], peer=task["peer"])
            params = GenParams(temperature=0.7, top_p=0.9, max_tokens=180)
            req = CompletionRequest(model=task["model"], prompt=prompt, params=params)
            out = await client.generate(req)
            r.rpush(OUT_Q, orjson.dumps({"id": task.get("id"), "critique": out.text}))
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(debate_loop())


