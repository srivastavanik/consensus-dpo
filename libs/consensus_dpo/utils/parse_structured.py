from __future__ import annotations

from typing import Any, Dict, Optional

from .json_utils import extract_json_object


def parse_generator_json(text: str) -> Optional[Dict[str, Any]]:
    obj = extract_json_object(text)
    if not obj:
        return None
    # normalize fields
    out = {
        "answer": obj.get("answer", ""),
        "rationale": obj.get("rationale", ""),
        "citations": obj.get("citations", []) or [],
    }
    if not isinstance(out["citations"], list):
        out["citations"] = [str(out["citations"])]
    return out


