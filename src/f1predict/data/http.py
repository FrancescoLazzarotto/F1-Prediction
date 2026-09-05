"""Shared HTTP client: token-bucket rate limiting, retries, disk cache.

Training walks every round of every season, which is hundreds of API calls.
Without a disk cache each re-train re-downloads the whole calendar; without a
rate limiter Jolpica starts returning 429s halfway through. This module gives
both, so callers can stay naive.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

_UA = "f1predict/3.0 (+https://github.com/f1predict)"


class RateLimiter:
    """Minimum-interval limiter, safe to share across threads."""

    def __init__(self, calls_per_second: float):
        self._min_interval = 1.0 / calls_per_second if calls_per_second > 0 else 0.0
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_allowed = now + self._min_interval


class JsonCache:
    """Content-addressed JSON cache on disk with per-entry TTL.

    A TTL of ``None`` means "never expires", which is correct for a completed
    season: those results cannot change.
    """

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return self.dir / f"{digest}.json"

    def get(self, key: str, ttl_s: float | None) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        if ttl_s is not None and (time.time() - path.stat().st_mtime) > ttl_s:
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            log.debug("Discarding corrupt cache entry %s", path.name)
            path.unlink(missing_ok=True)
            return None

    def put(self, key: str, value: Any) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(value, fh)
            tmp.replace(path)  # atomic, so a crash never leaves half a file
        except OSError as exc:
            log.debug("Could not write cache entry %s: %s", path.name, exc)
            tmp.unlink(missing_ok=True)


class HttpJsonClient:
    """JSON-over-HTTP client with retries, backoff and an optional disk cache."""

    def __init__(
        self,
        base_url: str,
        cache_dir: str | Path | None = None,
        rate_limit: float = 3.0,
        timeout_s: int = 20,
        max_retries: int = 4,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._limiter = RateLimiter(rate_limit)
        self._cache = JsonCache(cache_dir) if cache_dir else None
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _UA, "Accept": "application/json"})

    def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        ttl_s: float | None = None,
    ) -> dict:
        """GET ``path`` and return parsed JSON, serving from cache when fresh.

        Args:
            ttl_s: Cache lifetime. ``None`` caches forever (immutable history);
                pass a number for data that can still change.
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        key = f"{url}?{sorted((params or {}).items())}"

        if self._cache is not None and (hit := self._cache.get(key, ttl_s)) is not None:
            return hit

        payload = self._request(url, params)

        if self._cache is not None:
            self._cache.put(key, payload)
        return payload

    def _request(self, url: str, params: dict[str, Any] | None) -> dict:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self._limiter.acquire()
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout_s)
                if resp.status_code == 429:
                    wait = _retry_after(resp, attempt)
                    log.warning("Rate-limited by %s, waiting %.1fs", url, wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    backoff = min(2.0**attempt, 8.0)
                    log.debug("Request failed (%s), retrying in %.1fs", exc, backoff)
                    time.sleep(backoff)

        raise NetworkError(f"GET {url} failed after {self.max_retries} attempts: {last_exc}")


def _retry_after(resp: requests.Response, attempt: int) -> float:
    """Honour a Retry-After header, else exponential backoff."""
    header = resp.headers.get("Retry-After")
    if header:
        try:
            return min(float(header), 30.0)
        except ValueError:
            pass
    return min(2.0 ** (attempt + 1), 30.0)


class NetworkError(RuntimeError):
    """Raised when a remote source is unreachable after every retry.

    Callers catch this specifically to fall back to offline data, rather than
    swallowing every exception and masking real bugs.
    """
