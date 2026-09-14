"""
TraceNet v2 — Rate Limiter
===========================
Simple sliding-window rate limiter. No external dependencies.
Tracks requests per IP within a configurable time window.
"""

import time
from collections import defaultdict


class RateLimiter:
    """
    Sliding-window rate limiter keyed by client IP.

    Default: 60 requests per minute per IP.
    Scoring endpoints use a tighter limit (30/min) configured at call-site.
    """

    def __init__(self, max_calls: int = 60, window_seconds: int = 60):
        self.max_calls = max_calls
        self.window = window_seconds
        self._store: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str) -> tuple[bool, int, int]:
        """
        Returns (allowed, remaining, retry_after_seconds).
        allowed=False means the caller should receive 429.
        """
        now = time.time()
        # Prune expired timestamps
        self._store[key] = [t for t in self._store[key] if now - t < self.window]

        count = len(self._store[key])
        if count >= self.max_calls:
            oldest = self._store[key][0]
            retry_after = int(self.window - (now - oldest)) + 1
            return False, 0, retry_after

        self._store[key].append(now)
        remaining = self.max_calls - count - 1
        return True, remaining, 0

    def cleanup(self) -> int:
        """Prune fully-expired keys. Returns number of keys removed."""
        now = time.time()
        expired = [
            k for k, calls in self._store.items()
            if not any(now - t < self.window for t in calls)
        ]
        for k in expired:
            del self._store[k]
        return len(expired)
