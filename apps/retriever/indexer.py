from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


@dataclass
class Doc:
    doc_id: str
    title: str
    url: str
    text: str


def build_index(docs: Iterable[Doc], index_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
    if SentenceTransformer is None or faiss is None:
        raise RuntimeError("Retrieval dependencies missing. Install extras: pip install -e .[retrieval]")
    model = SentenceTransformer(model_name)
    chunks: List[Doc] = list(docs)
    embeddings = model.encode([d.text for d in chunks], show_progress_bar=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.asarray(embeddings, dtype=np.float32))
    Path(os.path.dirname(index_path)).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    meta_path = index_path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump([d.__dict__ for d in chunks], f)



