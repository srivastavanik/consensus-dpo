from __future__ import annotations

import json
import os
from typing import Dict, List

import datasets as hf
import requests

from libs.consensus_dpo.scoring.metrics import exact_match, average
from .select_chosen import pick_chosen_text


def load_gsm8k(split: str = "test", limit: int | None = 100) -> List[Dict]:
    ds = hf.load_dataset("gsm8k", "main", split=split)
    items = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        items.append({"prompt": row["question"], "gold": row["answer"]})
    return items


def eval_gsm8k(orchestrator_url: str, limit: int | None = 100) -> Dict[str, float]:
    data = load_gsm8k(limit=limit)
    em = []
    for item in data:
        resp = requests.post(orchestrator_url.rstrip("/") + "/consensus", json={
            "prompt": item["prompt"],
            "model": "gpt-oss-small",
            "k": 3,
            "m": 2,
            "r": 1,
        })
        if resp.status_code != 200:
            continue
        payload = resp.json()
        final = payload.get("final", {})
        pred = pick_chosen_text(final, payload.get("a", ""), payload.get("b", ""))
        em.append(exact_match(pred, item["gold"]))
    return {"GSM8K_EM": average(em)}


if __name__ == "__main__":
    url = os.getenv("ORCH_URL", "http://127.0.0.1:8000")
    print(eval_gsm8k(url, limit=20))


