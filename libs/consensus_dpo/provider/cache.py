from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class SqliteCache:
    """SQLite-backed cache keyed by hash(prompt, params).

    Stores JSON blobs with optional TTL.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(os.path.dirname(db_path)).mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expiry INTEGER
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _hash_key(prompt: str, params: Dict[str, Any]) -> str:
        payload = json.dumps({"prompt": prompt, "params": params}, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()

    def get(self, prompt: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        key = self._hash_key(prompt, params)
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT value, expiry FROM cache WHERE key = ?", (key,)).fetchone()
            if not row:
                return None
            value_str, expiry = row
            if expiry is not None and expiry < int(time.time()):
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
                return None
            return json.loads(value_str)
        finally:
            conn.close()

    def set(self, prompt: str, params: Dict[str, Any], value: Dict[str, Any], ttl_seconds: Optional[int] = None) -> None:
        key = self._hash_key(prompt, params)
        expiry = int(time.time()) + ttl_seconds if ttl_seconds else None
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "REPLACE INTO cache(key, value, expiry) VALUES (?, ?, ?)",
                (key, json.dumps(value, ensure_ascii=False), expiry),
            )
            conn.commit()
        finally:
            conn.close()


