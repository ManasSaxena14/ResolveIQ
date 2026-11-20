"""Rate limiting and optional API keys for the public API.

Every API call is counted against an identity over a sliding one-minute window:

- Anonymous callers (including the web app) are identified by client address and get
  ``anonymous_per_minute`` requests.
- Callers that send a valid key in the ``X-API-Key`` header are identified by that key
  and get ``key_per_minute`` requests. Keys come from the ``RESOLVEIQ_API_KEYS``
  environment variable, comma separated, and are never written to logs.
- A key that is sent but unknown is rejected with 401 rather than silently downgraded,
  so a typo in a client is noticed.

State lives in process memory, which suits a single-process demo server. Several
server processes would each keep their own counts.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from resolveiq.config import RateLimitConfig

API_KEY_HEADER = "X-API-Key"
KEYS_ENV_VAR = "RESOLVEIQ_API_KEYS"
WINDOW_SECONDS = 60.0
MAX_IDENTITIES = 10_000


@dataclass(frozen=True)
class Decision:
    allowed: bool
    limit: int
    remaining: int
    retry_after: int  # seconds until the oldest counted request leaves the window

    def headers(self) -> dict[str, str]:
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
        }
        if not self.allowed:
            headers["Retry-After"] = str(self.retry_after)
        return headers


class SlidingWindowLimiter:
    """Counts requests per identity over the last ``window`` seconds."""

    def __init__(
        self,
        window: float = WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        max_identities: int = MAX_IDENTITIES,
    ) -> None:
        self.window = window
        self.clock = clock
        self.max_identities = max_identities
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def hit(self, identity: str, limit: int) -> Decision:
        now = self.clock()
        with self._lock:
            hits = self._hits.pop(identity, None) or deque()
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            allowed = len(hits) < limit
            if allowed:
                hits.append(now)
            self._hits[identity] = hits  # most recently used last
            while len(self._hits) > self.max_identities:
                self._hits.popitem(last=False)
            retry = max(1, int(self.window - (now - hits[0])) + 1) if hits else 1
            return Decision(allowed, limit, max(0, limit - len(hits)), retry)


def load_api_keys(environ: dict[str, str] | None = None) -> frozenset[str]:
    raw = (environ if environ is not None else os.environ).get(KEYS_ENV_VAR, "")
    return frozenset(key.strip() for key in raw.split(",") if key.strip())


def key_fingerprint(key: str) -> str:
    """Short, non-reversible label for a key, used as its rate-limit identity."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


class AccessPolicy:
    """Decides who a request belongs to and whether it may proceed."""

    def __init__(
        self,
        config: RateLimitConfig,
        api_keys: Iterable[str] = (),
        limiter: SlidingWindowLimiter | None = None,
    ) -> None:
        self.config = config
        self.api_keys = frozenset(api_keys)
        self.limiter = limiter or SlidingWindowLimiter()

    @classmethod
    def from_config(cls, config: RateLimitConfig) -> AccessPolicy:
        return cls(config, load_api_keys())

    def is_valid_key(self, key: str) -> bool:
        return any(hmac.compare_digest(key, known) for known in self.api_keys)

    def client_address(self, peer: str | None, forwarded_for: str | None) -> str:
        if self.config.trust_forwarded_for and forwarded_for:
            return forwarded_for.split(",")[0].strip() or (peer or "unknown")
        return peer or "unknown"

    def check(
        self, api_key: str | None, peer: str | None, forwarded_for: str | None = None
    ) -> Decision | None:
        """Return the rate-limit decision, or None when the key is invalid (401)."""
        if api_key:
            if not self.is_valid_key(api_key):
                return None
            identity = f"key:{key_fingerprint(api_key)}"
            limit = self.config.key_per_minute
        else:
            identity = f"ip:{self.client_address(peer, forwarded_for)}"
            limit = self.config.anonymous_per_minute
        if not self.config.enabled:
            return Decision(True, limit, limit, 0)
        return self.limiter.hit(identity, limit)
