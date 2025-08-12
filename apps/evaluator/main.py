from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if pred.strip() == gold.strip() else 0.0


def evaluate_predictions(pairs_path: str) -> Dict[str, float]:
    ems: List[float] = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            # If eval set encodes gold under tools or prompt, adapt accordingly
            gold = row.get("gold", "")
            pred = row.get("chosen", {}).get("answer", "")
            if gold:
                ems.append(exact_match(pred, gold))
    return {"EM": float(np.mean(ems)) if ems else 0.0}


if __name__ == "__main__":
    path = os.environ.get("EVAL_PAIRS_PATH", "./data/pairs.dev.jsonl")
    print(evaluate_predictions(path))


