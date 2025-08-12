from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from a text and parse it.

    Returns None if parsing fails.
    """
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fallback: locate first {...}
    m = JSON_OBJECT_RE.search(text)
    if not m:
        return None
    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        return None


