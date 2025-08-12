from __future__ import annotations

import json
import os
from typing import Dict, List

import datasets as hf
import requests

from libs.consensus_dpo.scoring.metrics import average


def load_mmlu(split: str = "test", limit: int | None = 100) -> List[Dict]:
    ds = hf.load_dataset("cais/mmlu", split=split)
    items = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        stem = row.get("question", "")
        choices = [row.get(f"choices.{j}", "") for j in range(4)]
        gold_idx = int(row.get("answer", 0))
        gold = choices[gold_idx] if 0 <= gold_idx < len(choices) else ""
        items.append({"prompt": stem + "\n" + "\n".join(choices), "gold": gold})
    return items


def eval_mmlu(orchestrator_url: str, limit: int | None = 100) -> Dict[str, float]:
    data = load_mmlu(limit=limit)
    # Placeholder: compute a simple overlap score between final JSON string and gold text
    scores = []
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
        final = json.dumps(resp.json().get("final", {})).lower()
        scores.append(1.0 if item["gold"].lower() in final else 0.0)
    return {"MMLU_acc": average(scores)}


if __name__ == "__main__":
    url = os.getenv("ORCH_URL", "http://127.0.0.1:8000")
    print(eval_mmlu(url, limit=50))


