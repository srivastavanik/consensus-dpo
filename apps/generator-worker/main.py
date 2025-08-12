from __future__ import annotations

import asyncio
import json
import os
from typing import Dict

import orjson
import redis

from libs.consensus_dpo.provider import CompletionRequest, GenParams, NovitaClient


QUEUE_IN = os.getenv("GENERATOR_QUEUE_IN", "queue:generator:in")
QUEUE_OUT = os.getenv("GENERATOR_QUEUE_OUT", "queue:generator:out")


async def worker_loop() -> None:
    r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    client = NovitaClient()
    try:
        while True:
            _, payload = r.blpop(QUEUE_IN)
            task: Dict = json.loads(payload)
            params = GenParams(
                temperature=task.get("temperature", 0.8),
                top_p=task.get("top_p", 0.9),
                max_tokens=task.get("max_tokens", 512),
                seed=task.get("seed"),
            )
            req = CompletionRequest(model=task["model"], prompt=task["prompt"], params=params)
            out = await client.generate(req)
            r.rpush(QUEUE_OUT, orjson.dumps({"id": task.get("id"), "text": out.text, "usage": out.usage}))
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(worker_loop())


