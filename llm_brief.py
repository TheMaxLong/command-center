"""
PALM COMMAND — Morning Briefing generator via local Ollama LLM.

Polls events + alerts + feeds, builds a compact structured input, asks the
local LLM (default qwen3:14b on host:11434) to summarize as a brief.

Why: Max often comes off shift to a wall of numbers. A human-readable narrative
of the last 12h surfaces what actually matters faster than scanning tabs.

Privacy: stays on the box. No external API, no telemetry. Brief is cached
locally and re-generated on demand or every TTL_BRIEF seconds.

Endpoint wiring lives in camera_watcher.py — this module is pure logic so it's
trivially importable / testable.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any

# Default host is Mac's Ollama, reachable from inside the container.
OLLAMA_URL  = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")
# gemma4:e4b is faster and doesn't burn tokens inside <think> blocks.
# qwen3:14b is also installed — set OLLAMA_BRIEF_MODEL=qwen3:14b to use it
# (and bump NUM_PREDICT to ~2000 to allow for its thinking tokens).
OLLAMA_MODEL = os.environ.get("OLLAMA_BRIEF_MODEL", "gemma4:e4b")
NUM_PREDICT = int(os.environ.get("BRIEF_NUM_PREDICT", "1500"))
BRIEF_TTL   = int(os.environ.get("BRIEF_TTL", "900"))  # 15 min

_lock = threading.Lock()
_cache: dict = {"brief": None, "ts": 0.0, "model": None, "input_summary": None}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


SYSTEM_PROMPT = """You are a personal home-security briefing officer. The operator is a single user reading this in low light at 6am after an overnight shift.

Constraints:
- Under 180 words total.
- No filler, no preambles, no "based on the data".
- Use these exact section headers in this order, one short paragraph each:
  THREAT  — current aggregate threat tier and why
  PRESENCE — who was seen at the cameras (anonymous IDs / TRUSTED / UNKNOWN)
  WORLD   — outside-the-house feeds: fire, quake, weather, lightning, AQI
  ANOMALY — anything that doesn't fit the operator's normal rhythm
  RECOMMEND — at most one specific action item, or "no action" if nothing
- Tone: factual military brief. No "you should consider".
- Skip obvious normals. Don't repeat the data — interpret it.
- Mention specific times in 24h local (e.g. "21:46 doorbell motion").
- If a section has nothing, write "nothing notable" — don't pad."""


def _fetch_json(url: str, timeout: float = 4.0) -> Any:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PALM-COMMAND"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, json.JSONDecodeError, OSError):
        return None


def _gather_intel() -> dict:
    """Snapshot the data the LLM should reason over. Calls in-process modules
    directly — the previous HTTP-loopback approach deadlocked because the
    handler thread can't call back to its own (non-threaded) server."""
    out: dict[str, Any] = {}

    try:
        import intel_feeds
    except ImportError:
        return out

    feeds = intel_feeds.get_all_feeds()
    out["threat_level"] = feeds.get("threat_level")
    out["threat_label"] = feeds.get("threat_label")
    out["severity_counts"] = feeds.get("severity_counts")
    out["category_breakdown"] = feeds.get("category_breakdown")
    out["nearest_threat"] = feeds.get("nearest_threat")

    try:
        import intel_engine
        briefing = intel_engine.daily_briefing(None) or {}
        out["period_start"] = briefing.get("period_start")
        out["period_end"] = briefing.get("period_end")
        out["total_events"] = briefing.get("total_events")
        out["unique_cameras"] = briefing.get("unique_cameras")
        out["persons_seen"] = briefing.get("persons_seen")
        out["hourly_distribution"] = briefing.get("hourly_distribution")
        alerts = intel_engine.active_alerts(None) or []
        out["recent_alerts"] = alerts[:5] if isinstance(alerts, list) else []
    except (ImportError, Exception):
        pass

    try:
        out["aqi"] = intel_feeds.get_aqi_summary()
    except (AttributeError, Exception):
        out["aqi"] = {"status": "unavailable"}

    try:
        ls = intel_feeds.lightning_summary()
        out["lightning"] = {
            "strike_count": ls.get("strike_count"),
            "nearest_km": ls.get("nearest_distance_km"),
            "mqtt_connected": (ls.get("mqtt_status") or {}).get("connected"),
        }
    except (AttributeError, Exception):
        out["lightning"] = {"status": "unavailable"}

    return out


def _call_ollama(intel: dict) -> dict:
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": "Current snapshot to brief on:\n\n" + json.dumps(intel, indent=2, default=str),
        "stream": False,
        "options": {
            "temperature": 0.4,
            "num_predict": NUM_PREDICT,
        },
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def generate_morning_brief(force: bool = False) -> dict:
    """Returns {brief, model, generated_at, ttl_remaining, input_summary, ...}."""
    now = time.time()
    with _lock:
        cached = _cache.copy()
    if not force and cached.get("brief") and (now - cached.get("ts", 0)) < BRIEF_TTL:
        return {
            "brief": cached["brief"],
            "model": cached["model"],
            "generated_at": cached["ts"],
            "ttl_remaining": int(BRIEF_TTL - (now - cached["ts"])),
            "input_summary": cached.get("input_summary"),
            "cached": True,
        }

    intel = _gather_intel()
    if not intel.get("threat_level"):
        return {"error": "intel_feeds offline (no /feeds response)", "cached": False}

    try:
        resp = _call_ollama(intel)
    except (urllib.error.URLError, OSError) as e:
        return {"error": f"ollama unreachable: {e}", "model": OLLAMA_MODEL, "cached": False}
    except json.JSONDecodeError as e:
        return {"error": f"ollama returned non-json: {e}", "cached": False}

    raw = resp.get("response", "") or ""
    # Strip qwen3-style <think>...</think> blocks
    text = _THINK_RE.sub("", raw).strip()

    input_summary = {
        "threat_level": intel.get("threat_level"),
        "total_events": intel.get("total_events"),
        "persons_seen": len(intel.get("persons_seen") or []),
        "aqi_status": (intel.get("aqi") or {}).get("status"),
        "lightning_count": (intel.get("lightning") or {}).get("strike_count"),
    }

    with _lock:
        _cache.update({"brief": text, "ts": now, "model": OLLAMA_MODEL, "input_summary": input_summary})

    return {
        "brief": text,
        "model": OLLAMA_MODEL,
        "generated_at": now,
        "ttl_remaining": BRIEF_TTL,
        "input_summary": input_summary,
        "cached": False,
    }
