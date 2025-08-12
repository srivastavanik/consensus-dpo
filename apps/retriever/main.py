from __future__ import annotations

import os
from typing import List, Tuple

from fastapi import FastAPI

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # lazy import for environments without faiss


app = FastAPI(title="Consensus-DPO Retriever")

INDEX_PATH = os.getenv("INDEX_PATH", "./data/index/wiki.faiss")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "index": INDEX_PATH}


@app.get("/search")
async def search(q: str, k: int = 5) -> dict:
    # Placeholder API with TODO for real FAISS; keep interface stable
    # In full implementation, embed `q`, search FAISS, return doc IDs + scores
    return {"query": q, "results": [("doc:0", 0.0)] * k}


@app.get("/fetch")
async def fetch(doc_id: str) -> dict:
    # TODO: fetch from a local doc store; keep shape stable
    return {"doc_id": doc_id, "text": "", "title": "", "url": ""}


