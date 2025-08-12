from __future__ import annotations

import json
import os
from typing import Dict, List

import requests

from libs.consensus_dpo.scoring.metrics import exact_match, average


def eval_prompts(prompts: List[Dict], orchestrator_url: str) -> Dict[str, float]:
    ems = []
    for item in prompts:
        prompt = item["prompt"]
        gold = item.get("gold", "")
        resp = requests.post(orchestrator_url.rstrip("/") + "/consensus", json={
            "prompt": prompt,
            "model": item.get("model", "gpt-oss-small"),
            "k": item.get("k", 3),
            "m": item.get("m", 2),
            "r": item.get("r", 1),
        })
        if resp.status_code != 200:
            continue
        data = resp.json()
        final = data.get("final", {})
        # decode winner to pick chosen text; we stored pair separately, but here just EM vs gold
        pred = json.dumps(final)
        if gold:
            ems.append(exact_match(pred, gold))
    return {"EM": average(ems)}


if __name__ == "__main__":
    prompts_path = os.getenv("EVAL_PROMPTS", "./data/eval.prompts.jsonl")
    url = os.getenv("ORCH_URL", "http://127.0.0.1:8000")
    items: List[Dict] = []
    if os.path.exists(prompts_path):
        with open(prompts_path, "r", encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))
    print(eval_prompts(items, url))


