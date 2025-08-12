from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from .base import Completion, CompletionRequest, GenParams, ModelProvider
from .cache import SqliteCache
from .rate_limiter import TokenBucketLimiter


class NovitaConfig(BaseSettings):
    """Configuration for Novita provider (loads from process env and .env)."""

    # Load .env automatically
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    api_key: str = os.getenv("NOVITA_API_KEY", "")
    base_url: str = os.getenv("NOVITA_BASE_URL", "https://api.novita.ai")
    api_path: str = os.getenv("NOVITA_API_PATH", "/v1/chat/completions")
    requests_per_second: float = float(os.getenv("NOVITA_REQUESTS_PER_SECOND", 5))
    cache_db_path: str = os.getenv("RUNS_DIR", "./data/runs") + "/novita_cache.sqlite"


class _ChatMessage(BaseModel):
    role: str
    content: str


class NovitaClient(ModelProvider):
    """Novita AI GPT-OSS client (OpenAI-compatible by default).

    The client uses an async HTTPX session, token-bucket rate limiting, retry with jitter,
    and a SQLite cache keyed by (prompt, params).
    """

    def __init__(self, config: Optional[NovitaConfig] = None) -> None:
        self.config = config or NovitaConfig()
        self._client = httpx.AsyncClient(timeout=60)
        self._limiter = TokenBucketLimiter(self.config.requests_per_second)
        self._cache = SqliteCache(self.config.cache_db_path)

    async def _post_chat_completions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self.config.base_url.rstrip("/") + self.config.api_path
        api_key = (self.config.api_key or "").strip()
        if not api_key:
            raise ValueError(
                "NOVITA_API_KEY is not set. Define it in your environment or .env file."
            )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(5),
            wait=wait_exponential_jitter(initial=0.5, max=8),
            retry=retry_if_exception_type(httpx.HTTPError),
        ):
            with attempt:
                async with self._limiter():
                    resp = await self._client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                return resp.json()

    def _to_payload(self, req: CompletionRequest) -> Dict[str, Any]:
        params = req.params
        messages = [_ChatMessage(role="user", content=req.prompt).model_dump()]
        payload = {
            "model": req.model,
            "messages": messages,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
        }
        if params.seed is not None:
            payload["seed"] = params.seed
        if params.stop:
            payload["stop"] = params.stop
        return payload

    def _cache_key_params(self, req: CompletionRequest) -> Dict[str, Any]:
        return {
            "model": req.model,
            "temperature": req.params.temperature,
            "top_p": req.params.top_p,
            "max_tokens": req.params.max_tokens,
            "seed": req.params.seed,
            "stop": req.params.stop or [],
        }

    async def generate(self, req: CompletionRequest) -> Completion:
        cache_params = self._cache_key_params(req)
        cached = self._cache.get(req.prompt, cache_params)
        if cached:
            choice = cached["choices"][0]["message"]["content"]
            usage = cached.get("usage", {})
            return Completion(model=req.model, prompt=req.prompt, text=choice, usage=usage, raw=cached)

        payload = self._to_payload(req)
        raw = await self._post_chat_completions(payload)
        # Basic OpenAI-compatible shape
        text = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = raw.get("usage", {})
        # Cache short-lived to reduce retries during sweeps
        self._cache.set(req.prompt, cache_params, raw, ttl_seconds=600)
        return Completion(model=req.model, prompt=req.prompt, text=text, usage=usage, raw=raw)

    async def batchGenerate(self, reqs: List[CompletionRequest]) -> List[Completion]:  # noqa: N802
        return [await self.generate(r) for r in reqs]

    async def aclose(self) -> None:
        await self._client.aclose()


