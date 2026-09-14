"""
TraceNet v2 — Session Persistence
===================================
JSON-file-backed session state.

On Railway free tier the filesystem is ephemeral (resets on restart),
so stats still reset on redeploy — but within a running session they
survive code-level exceptions and are readable by admin endpoints.

To upgrade to true persistence: swap _load/_save for SQLite calls.
The SessionState public interface stays identical.
"""

import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

STATE_FILE = os.getenv("STATE_FILE", "models/session_state.json")


# ── Helpers ───────────────────────────────────────────────────────

def _default_stats() -> dict:
    return {
        "total_scored":  0,
        "auto_approved": 0,
        "human_review":  0,
        "auto_blocked":  0,
        "session_start": datetime.now(timezone.utc).isoformat(),
    }


def _load() -> dict:
    if not os.path.exists(STATE_FILE):
        return {"stats": _default_stats(), "sars": {}}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"State file unreadable ({exc}). Starting fresh.")
        return {"stats": _default_stats(), "sars": {}}


def _save(state: dict) -> None:
    try:
        os.makedirs(os.path.dirname(STATE_FILE) or ".", exist_ok=True)
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)
    except OSError as exc:
        logger.error(f"State save failed: {exc}")


# ── Public class ──────────────────────────────────────────────────

class SessionState:
    """
    Centralised session state — stats counters + SAR cache.
    Loaded once at startup via lifespan and injected into api.py globals.
    """

    def __init__(self) -> None:
        raw = _load()
        self.stats: dict      = raw.get("stats", _default_stats())
        self.sars:  dict[str, dict] = raw.get("sars", {})
        self._counter: int    = len(self.sars)
        logger.info(
            f"SessionState ready — "
            f"{self.stats['total_scored']} txns scored, {self._counter} SARs"
        )

    # ── Stats ─────────────────────────────────────────────────────

    def increment(self, verdict: str) -> None:
        self.stats["total_scored"] += 1
        field = {
            "AUTO_APPROVE": "auto_approved",
            "HUMAN_REVIEW": "human_review",
            "AUTO_BLOCK":   "auto_blocked",
        }.get(verdict, "auto_approved")
        self.stats[field] += 1
        _save({"stats": self.stats, "sars": self.sars})

    def get_stats(self) -> dict:
        return {**self.stats}

    def reset_stats(self) -> None:
        """Reset counters only — preserves SAR history."""
        self.stats = _default_stats()
        _save({"stats": self.stats, "sars": self.sars})

    # ── SAR cache ─────────────────────────────────────────────────

    def next_sar_id(self) -> str:
        self._counter += 1
        year = datetime.now().year
        return f"SAR-{year}-{self._counter:04d}"

    def add_sar(self, tx_hash: str, report: dict) -> None:
        self.sars[tx_hash] = report
        _save({"stats": self.stats, "sars": self.sars})

    def get_sar(self, tx_hash: str) -> dict | None:
        return self.sars.get(tx_hash)

    def list_sars(self) -> list[dict]:
        return [
            {
                "report_id":    v["report_id"],
                "tx_hash":      k,
                "risk_percent": v["risk_percent"],
                "verdict":      v.get("verdict", "AUTO_BLOCK"),
                "pattern":      v.get("pattern", "unknown"),
                "generated_at": v["generated_at"],
            }
            for k, v in self.sars.items()
        ]

    def sar_exists(self, tx_hash: str) -> bool:
        return tx_hash in self.sars
