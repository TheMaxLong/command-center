#!/usr/bin/env python3.12
"""
PALM COMMAND — Push Notification Engine

Delivers real-time alerts to your phone/devices via ntfy.sh (no account
required) and optional Twilio SMS fallback.

Configuration (environment variables):
  NTFY_TOPIC    — ntfy topic slug, e.g. "palm-command-home" (REQUIRED to enable push)
  NTFY_URL      — ntfy server, default "https://ntfy.sh"
  NTFY_TOKEN    — auth token for private ntfy topics (optional)
  TWILIO_SID    — Twilio Account SID (optional SMS fallback)
  TWILIO_TOKEN  — Twilio Auth Token
  TWILIO_FROM   — Twilio sender number "+1..."
  TWILIO_TO     — Your phone number "+1..."

Severity → ntfy priority:
  critical  → 5  (urgent — breaks through Do Not Disturb)
  high      → 4  (high)
  medium    → 3  (default)
  low       → 2  (low)
  info      → 1  (min)

Rate limiting: same (type, camera) pair silenced for COOLDOWN_SEC after
first delivery to prevent alert storms.
"""
from __future__ import annotations

import os, threading, time, urllib.request, urllib.error
from collections import defaultdict
from typing import Optional

# ── Config ────────────────────────────────────────────────────────
NTFY_TOPIC  = os.environ.get("NTFY_TOPIC", "")
NTFY_URL    = os.environ.get("NTFY_URL",   "https://ntfy.sh").rstrip("/")
NTFY_TOKEN  = os.environ.get("NTFY_TOKEN", "")

TWILIO_SID   = os.environ.get("TWILIO_SID",   "")
TWILIO_TOKEN = os.environ.get("TWILIO_TOKEN", "")
TWILIO_FROM  = os.environ.get("TWILIO_FROM",  "")
TWILIO_TO    = os.environ.get("TWILIO_TO",    "")

COOLDOWN_SEC = int(os.environ.get("NTFY_COOLDOWN", "300"))   # 5 min default

_PRIORITY = {
    "critical": 5,
    "high":     4,
    "medium":   3,
    "low":      2,
    "info":     1,
}

_SEVERITY_EMOJI = {
    "critical": "🚨",
    "high":     "⚠️",
    "medium":   "🔔",
    "low":      "ℹ️",
    "info":     "📡",
}

# ── Runtime state ─────────────────────────────────────────────────
_lock       = threading.Lock()
_cooldowns: dict[str, float]  = {}   # key → last_delivered_ts
_stats: dict[str, int]        = defaultdict(int)
_delivery_log: list[dict]     = []   # last 50 deliveries
_MAX_LOG = 50


def _cooldown_key(type_: str, camera_id: Optional[str]) -> str:
    return f"{type_}::{camera_id or '*'}"


def _is_on_cooldown(type_: str, camera_id: Optional[str]) -> bool:
    key = _cooldown_key(type_, camera_id)
    with _lock:
        last = _cooldowns.get(key, 0)
        return (time.time() - last) < COOLDOWN_SEC


def _set_cooldown(type_: str, camera_id: Optional[str]) -> None:
    key = _cooldown_key(type_, camera_id)
    with _lock:
        _cooldowns[key] = time.time()


def _log_delivery(type_: str, severity: str, message: str,
                  channel: str, ok: bool, error: str = "") -> None:
    entry = {
        "ts":       time.time(),
        "type":     type_,
        "severity": severity,
        "message":  message[:120],
        "channel":  channel,
        "ok":       ok,
        "error":    error[:120] if error else "",
    }
    with _lock:
        _delivery_log.append(entry)
        if len(_delivery_log) > _MAX_LOG:
            _delivery_log.pop(0)
        _stats["total"] += 1
        if ok:
            _stats["delivered"] += 1
        else:
            _stats["failed"] += 1


# ── ntfy.sh delivery ──────────────────────────────────────────────

def _send_ntfy(type_: str, severity: str, message: str,
               camera_id: Optional[str], title: str) -> bool:
    if not NTFY_TOPIC:
        return False

    priority = str(_PRIORITY.get(severity.lower(), 3))
    topic    = NTFY_TOPIC
    url      = f"{NTFY_URL}/{topic}"
    tags     = ["palm-command", type_.replace("_", "-")]
    if camera_id:
        tags.append(f"cam-{camera_id}")

    headers: dict[str, str] = {
        "Title":    title,
        "Priority": priority,
        "Tags":     ",".join(tags),
        "Content-Type": "text/plain",
    }
    if NTFY_TOKEN:
        headers["Authorization"] = f"Bearer {NTFY_TOKEN}"

    body = message.encode("utf-8")
    req  = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            ok = r.status in (200, 201, 204)
            _log_delivery(type_, severity, message, "ntfy", ok)
            return ok
    except Exception as e:
        _log_delivery(type_, severity, message, "ntfy", False, str(e))
        return False


# ── Twilio SMS delivery ───────────────────────────────────────────

def _send_sms(type_: str, severity: str, message: str,
              camera_id: Optional[str], title: str) -> bool:
    if not all([TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, TWILIO_TO]):
        return False

    body_text = f"PALM COMMAND [{severity.upper()}] {title}: {message}"[:1600]
    import base64, urllib.parse
    auth_str = base64.b64encode(f"{TWILIO_SID}:{TWILIO_TOKEN}".encode()).decode()
    url  = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json"
    data = urllib.parse.urlencode({
        "From": TWILIO_FROM,
        "To":   TWILIO_TO,
        "Body": body_text,
    }).encode()
    req  = urllib.request.Request(url, data=data,
                                  headers={"Authorization": f"Basic {auth_str}"},
                                  method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            ok = r.status in (200, 201)
            _log_delivery(type_, severity, message, "sms", ok)
            return ok
    except Exception as e:
        _log_delivery(type_, severity, message, "sms", False, str(e))
        return False


# ── Public API ────────────────────────────────────────────────────

def notify(
    type_: str,
    severity: str,
    message: str,
    camera_id: Optional[str] = None,
    title: Optional[str]     = None,
    force: bool              = False,
) -> None:
    """
    Fire-and-forget alert delivery. Returns immediately; delivery runs in a
    background thread. Respects cooldown unless force=True.

    Args:
        type_:     Alert category (e.g. "face_intel", "threat_scenario", "motion")
        severity:  One of "critical" | "high" | "medium" | "low" | "info"
        message:   Alert body text (max ~500 chars recommended)
        camera_id: Optional camera that triggered the alert
        title:     Notification title; defaults to "PALM COMMAND — <type>"
        force:     Skip cooldown check (use for critical one-shot alerts)
    """
    if not NTFY_TOPIC and not all([TWILIO_SID, TWILIO_TOKEN]):
        return   # no delivery channels configured — silently skip

    if not force and _is_on_cooldown(type_, camera_id):
        return   # rate-limited

    _set_cooldown(type_, camera_id)

    emoji = _SEVERITY_EMOJI.get(severity.lower(), "📡")
    if title is None:
        cam_tag = f" [{camera_id}]" if camera_id else ""
        title   = f"{emoji} PALM COMMAND{cam_tag} — {type_.replace('_', ' ').upper()}"

    def _deliver():
        ntfy_ok = _send_ntfy(type_, severity, message, camera_id, title)
        # SMS fallback: only when ntfy fails AND severity is high/critical
        if not ntfy_ok and severity.lower() in ("critical", "high"):
            _send_sms(type_, severity, message, camera_id, title)

    threading.Thread(target=_deliver, daemon=True).start()


def notify_high(type_: str, message: str, camera_id: Optional[str] = None,
                title: Optional[str] = None) -> None:
    """Convenience wrapper for HIGH severity."""
    notify(type_, "high", message, camera_id, title)


def notify_critical(type_: str, message: str, camera_id: Optional[str] = None,
                    title: Optional[str] = None) -> None:
    """Convenience wrapper for CRITICAL severity — also triggers SMS."""
    notify(type_, "critical", message, camera_id, title, force=True)


def test_notify() -> dict:
    """Send a test notification to verify delivery channels are configured."""
    notify(
        type_="system_test",
        severity="info",
        message="PALM COMMAND notification test — delivery confirmed.",
        title="📡 PALM COMMAND — SYSTEM TEST",
        force=True,
    )
    return {
        "sent":         True,
        "ntfy_enabled": bool(NTFY_TOPIC),
        "ntfy_url":     f"{NTFY_URL}/{NTFY_TOPIC}" if NTFY_TOPIC else None,
        "sms_enabled":  bool(TWILIO_SID and TWILIO_TOKEN),
    }


def get_status() -> dict:
    """Return notifier configuration and delivery statistics."""
    with _lock:
        stats = dict(_stats)
        log   = list(reversed(_delivery_log))[:20]
        n_cooldowns = len(_cooldowns)
    return {
        "ntfy": {
            "enabled": bool(NTFY_TOPIC),
            "topic":   NTFY_TOPIC or None,
            "url":     f"{NTFY_URL}/{NTFY_TOPIC}" if NTFY_TOPIC else None,
            "has_token": bool(NTFY_TOKEN),
        },
        "sms": {
            "enabled": bool(TWILIO_SID and TWILIO_TOKEN),
            "from":    TWILIO_FROM or None,
            "to":      TWILIO_TO   or None,
        },
        "cooldown_sec":   COOLDOWN_SEC,
        "active_cooldowns": n_cooldowns,
        "stats":          stats,
        "recent_deliveries": log,
    }


def briefing() -> str:
    """Return a plain-text status summary for the PALANTIR terminal."""
    st = get_status()
    lines = ["▸ NOTIFICATION ENGINE — PALM COMMAND"]
    if st["ntfy"]["enabled"]:
        lines.append(f"▸ ntfy.sh active → {st['ntfy']['url']}")
    else:
        lines.append("▸ ntfy.sh DISABLED  (set NTFY_TOPIC env var to enable)")
    if st["sms"]["enabled"]:
        lines.append(f"▸ SMS active → {st['sms']['to']}")
    else:
        lines.append("▸ SMS DISABLED  (set TWILIO_SID/TOKEN/FROM/TO to enable)")
    stats = st["stats"]
    if stats:
        lines.append(f"▸ Delivered {stats.get('delivered', 0)} / {stats.get('total', 0)} "
                     f"({stats.get('failed', 0)} failed)")
    recent = st["recent_deliveries"][:3]
    for r in recent:
        ts  = r.get("ts", 0)
        ts_str = ""
        try:
            from datetime import datetime
            ts_str = datetime.fromtimestamp(ts).strftime("%H:%M")
        except Exception:
            pass
        ok_str = "✓" if r.get("ok") else "✗"
        lines.append(f"  {ok_str} [{ts_str}] {r.get('severity','').upper()} {r.get('type','')} "
                     f"via {r.get('channel','')}")
    return "\n".join(lines)
