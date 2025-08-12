from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class GenParams:
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 512
    seed: Optional[int] = None
    stop: Optional[List[str]] = None


@dataclass
class Completion:
    model: str
    prompt: str
    text: str
    usage: Dict[str, Any]
    raw: Dict[str, Any]


@dataclass
class CompletionRequest:
    model: str
    prompt: str
    params: GenParams


class ModelProvider(abc.ABC):
    """Abstract provider interface for text generation.

    Implementations should be stateless and concurrency-safe. Any rate limiting
    or caching should be handled externally or via composition.
    """

    @abc.abstractmethod
    async def generate(self, req: CompletionRequest) -> Completion:  # pragma: no cover
        raise NotImplementedError

    @abc.abstractmethod
    async def batchGenerate(self, reqs: List[CompletionRequest]) -> List[Completion]:  # noqa: N802
        raise NotImplementedError

    async def embeddings(self, texts: List[str]) -> List[List[float]]:  # pragma: no cover
        raise NotImplementedError


