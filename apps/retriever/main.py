from __future__ import annotations

import os
from typing import List, Tuple

from fastapi import FastAPI
import json
import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # lazy import for environments without faiss


app = FastAPI(title="Consensus-DPO Retriever")

INDEX_PATH = os.getenv("INDEX_PATH", "./data/index/wiki.faiss")
META_PATH = INDEX_PATH + ".meta.json"


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "index": INDEX_PATH}


@app.get("/search")
async def search(q: str, k: int = 5) -> dict:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import faiss  # type: ignore
    except Exception:
        return {"query": q, "results": []}
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        return {"query": q, "results": []}
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode([q], normalize_embeddings=True)
    index = faiss.read_index(INDEX_PATH)
    D, I = index.search(np.asarray(emb, dtype=np.float32), k)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    results = []
    for idx, score in zip(I[0].tolist(), D[0].tolist()):
        if idx < 0 or idx >= len(meta):
            continue
        m = meta[idx]
        results.append(((m.get("doc_id") or str(idx)), float(score)))
    return {"query": q, "results": results}


@app.get("/fetch")
async def fetch(doc_id: str) -> dict:
    # TODO: fetch from a local doc store; keep shape stable
    return {"doc_id": doc_id, "text": "", "title": "", "url": ""}


