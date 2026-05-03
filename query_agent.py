#!/usr/bin/env python3.12
"""
PALM COMMAND — Integrated Security AI Agent.

A rule-based + optional LLM query engine you can talk to directly.
Understands natural language questions about the security system and
returns structured, mission-briefing style responses.

Optionally uses an LLM if OPENAI_API_KEY or ANTHROPIC_API_KEY is set.
Falls back to the rule-based engine when no API key is present — fully
self-contained, no internet required.

Intents handled:
  summary         — "what happened today", "give me a briefing"
  who_today       — "who was here today/tonight/yesterday"
  recent_events   — "what's been happening", "last 10 events"
  person_info     — "tell me about REGULAR-001", "who is profile 3"
  stranger_check  — "any strangers?", "unknown visitors"
  anomaly_check   — "anything unusual?", "anomalies"
  time_query      — "what happened at 2am", "activity between noon and 3pm"
  count_query     — "how many people today", "events this week"
  watchlist_add   — "alert me when [condition]"
  velocity        — "are events increasing?", "trend"
  camera_compare  — "which camera is busiest"
  help            — "what can you do", "help"
"""
from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import event_db
import trend_analyzer

# ── Optional: intel feeds (loaded lazily to avoid startup errors) ──
_feeds_available = False
def _get_feeds():
    global _feeds_available
    try:
        import intel_feeds as _if
        _feeds_available = True
        return _if
    except Exception:
        return None

# ── Optional LLM backend ──────────────────────────────────────────
_LLM_CLIENT   = None
_LLM_PROVIDER = None

def _init_llm():
    global _LLM_CLIENT, _LLM_PROVIDER
    if _LLM_CLIENT is not None:
        return _LLM_CLIENT is not False

    oai_key = os.environ.get("OPENAI_API_KEY", "")
    ant_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if oai_key:
        try:
            from openai import OpenAI
            _LLM_CLIENT   = OpenAI(api_key=oai_key)
            _LLM_PROVIDER = "openai"
            print("[agent] LLM: OpenAI backend", flush=True)
            return True
        except ImportError:
            pass

    if ant_key:
        try:
            import anthropic
            _LLM_CLIENT   = anthropic.Anthropic(api_key=ant_key)
            _LLM_PROVIDER = "anthropic"
            print("[agent] LLM: Anthropic backend", flush=True)
            return True
        except ImportError:
            pass

    _LLM_CLIENT = False   # mark as "no LLM"
    return False


def _llm_query(question: str, context: str) -> Optional[str]:
    """
    Send question + context to the LLM. Returns response string or None.
    Context is a concise JSON/text dump of relevant DB data.
    """
    if not _init_llm():
        return None
    system = (
        "You are PALANTIR, a tactical AI security analyst for a home surveillance system. "
        "Respond in short, precise, military-report style. Use mission briefing format. "
        "Be direct and factual. Use data from the provided context. "
        "Address the user as 'OPERATOR'. Max 3 paragraphs."
    )
    prompt = f"CONTEXT:\n{context}\n\nOPERATOR QUERY: {question}"
    try:
        if _LLM_PROVIDER == "openai":
            resp = _LLM_CLIENT.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
            )
            return resp.choices[0].message.content
        elif _LLM_PROVIDER == "anthropic":
            resp = _LLM_CLIENT.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
    except Exception as e:
        print(f"[agent] LLM error: {e}", flush=True)
    return None


# ── In-memory watchlist ───────────────────────────────────────────
# Each rule: { "id": int, "condition": str, "raw": str, "created": float }
_watchlist: list[dict] = []
_watchlist_id = 0


def add_watchlist_rule(raw_text: str) -> dict:
    global _watchlist_id
    _watchlist_id += 1
    rule = {
        "id":        _watchlist_id,
        "raw":       raw_text.strip(),
        "condition": _extract_condition(raw_text),
        "created":   time.time(),
        "hits":      0,
    }
    _watchlist.append(rule)
    return rule


def get_watchlist() -> list[dict]:
    return list(_watchlist)


def _extract_condition(text: str) -> str:
    """Best-effort plain English condition normalisation."""
    text = text.lower()
    if "unknown" in text or "stranger" in text:
        return "unknown_person"
    if "vehicle" in text or "car" in text or "truck" in text:
        return "vehicle"
    if "person" in text or "people" in text:
        return "person"
    if "night" in text or "midnight" in text or "after dark" in text:
        return "off_hours"
    return "any_activity"


# ── Intent classification ─────────────────────────────────────────

_INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("summary",        ["briefing", "summary", "what happened", "update me", "status", "report", "overview"]),
    ("who_today",      ["who was here", "who came", "who visited", "who showed up", "who appeared"]),
    ("stranger_check", ["stranger", "unknown", "unfamiliar", "never seen", "new person", "new visitor"]),
    ("anomaly_check",  ["unusual", "anomaly", "anomalies", "weird", "abnormal", "suspicious", "odd"]),
    ("velocity",       ["trend", "increasing", "decreasing", "more activity", "velocity", "getting busier"]),
    ("camera_compare", ["which camera", "busiest cam", "most active camera", "camera comparison"]),
    ("count_query",    ["how many", "count", "number of", "total events", "total people"]),
    ("time_query",     ["at 12", "at 1", "at 2", "at 3", "at 4", "at 5", "at 6", "at 7", "at 8",
                        "at 9", "at 10", "at 11", "overnight", "midnight", "noon", "morning",
                        "afternoon", "evening", "night", "between", "from "]),
    ("recent_events",  ["recent", "latest", "last event", "last 5", "last 10", "what's been", "what has been"]),
    ("person_info",    ["tell me about", "who is", "profile", "regular-", "unknown-", "information on"]),
    ("watchlist_add",  ["alert me", "notify me", "watch for", "flag", "tell me when", "ping me"]),
    ("watchlist_show", ["watchlist", "rules", "active alerts", "my alerts", "what are you watching"]),
    # ── External intelligence feeds ──────────────────────────────────
    ("earthquake",     ["earthquake", "quake", "seismic", "tremor", "fault", "shaking", "richter", "magnitude"]),
    ("weather_alert",  ["weather alert", "fire weather", "red flag", "air quality", "aqi", "heat warning",
                        "wind advisory", "dust storm", "extreme heat", "nws alert", "haboob"]),
    ("fire_intel",     ["calfire", "wildfire", "brush fire", "structure fire", "fire near", "cal fire",
                        "active fire", "fire incident", "acres burning", "containment"]),
    ("local_incidents",["local crime", "citizen", "nearby incident", "neighborhood", "what's happening near",
                        "crime near", "shooting near", "robbery", "police near", "ems near",
                        "911 near", "incident near", "local incident", "area crime"]),
    ("area_threat",    ["threat level", "area threat", "threat assessment", "danger level", "safe",
                        "how safe", "area status", "local threat", "anything dangerous", "any danger"]),
    ("plates",         ["license plate", "plate", "lpr", "vehicle plate", "plates seen", "plate log"]),
    ("help",           ["help", "what can you do", "commands", "options", "capabilities"]),
]


def _classify_intent(text: str) -> str:
    t = text.lower()
    for intent, patterns in _INTENT_PATTERNS:
        if any(p in t for p in patterns):
            return intent
    return "general"


# ── Time range parsing ────────────────────────────────────────────

def _parse_time_range(text: str) -> tuple[float, float]:
    """Return (start_ts, end_ts) based on text like 'today', 'yesterday', 'last 24h', '2am'."""
    now = datetime.now(tz=timezone.utc)
    t   = text.lower()

    if "yesterday" in t:
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end   = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif "last week" in t or "this week" in t:
        start = now - timedelta(days=7)
        end   = now
    elif "last hour" in t:
        start = now - timedelta(hours=1)
        end   = now
    elif "overnight" in t or "midnight" in t:
        start = now.replace(hour=22, minute=0, second=0, microsecond=0) - timedelta(days=1)
        end   = now.replace(hour=6, minute=0, second=0, microsecond=0)
    elif "morning" in t:
        start = now.replace(hour=6, minute=0, second=0, microsecond=0)
        end   = now.replace(hour=12, minute=0, second=0, microsecond=0)
    elif "afternoon" in t:
        start = now.replace(hour=12, minute=0, second=0, microsecond=0)
        end   = now.replace(hour=18, minute=0, second=0, microsecond=0)
    elif "evening" in t:
        start = now.replace(hour=18, minute=0, second=0, microsecond=0)
        end   = now.replace(hour=22, minute=0, second=0, microsecond=0)
    else:
        # Try to extract an hour like "at 2am", "at 14:00", "between 3pm and 5pm"
        hours = re.findall(r'(\d{1,2})(?:am|pm)?', t)
        if hours:
            h = int(hours[0])
            if "pm" in t and h < 12:
                h += 12
            start = now.replace(hour=h, minute=0, second=0, microsecond=0)
            end   = start + timedelta(hours=1)
            if end > now:
                start -= timedelta(days=1)
                end   -= timedelta(days=1)
        else:
            # Default: today
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end   = now
    return start.timestamp(), end.timestamp()


# ── Profile helpers ───────────────────────────────────────────────

def _profiles_seen_since(cutoff_ts: float) -> list[dict]:
    """Return profiles that had at least one sighting after cutoff_ts."""
    profiles = event_db.get_all_profiles()
    import json as _json
    result = []
    for p in profiles:
        sightings = event_db.get_profile_sightings(p["id"])
        recent = [s for s in sightings if s["ts"] >= cutoff_ts]
        if recent:
            result.append({
                "id":        p["id"],
                "label":     p.get("label") or (f"REGULAR-{p['id']:03d}" if p["sightings"] >= 4 else f"UNKNOWN-{p['id']:03d}"),
                "sightings": p["sightings"],
                "cameras":   _json.loads(p["cameras"]),
                "last_seen": p["last_seen"],
                "today_count": len(recent),
                "is_regular": p["sightings"] >= 4,
            })
    return sorted(result, key=lambda x: -x["today_count"])


# ── Response formatting ───────────────────────────────────────────

def _fmtts(ts: float) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%a %b %-d · %H:%M")


def _fmt_profiles(profiles: list[dict]) -> str:
    if not profiles:
        return "No persons detected."
    lines = []
    for p in profiles:
        reg   = "REGULAR" if p["is_regular"] else "UNKNOWN"
        cams  = ", ".join(c.upper() for c in p.get("cameras", [])) or "unknown cam"
        lines.append(f"  [{reg}] {p['label']} — seen {p['today_count']}× on {cams}")
    return "\n".join(lines)


# ── Intent handlers ───────────────────────────────────────────────

def _handle_summary(text: str, camera_id: Optional[str]) -> dict:
    cutoff = (datetime.now(tz=timezone.utc) - timedelta(hours=24)).timestamp()
    events = event_db.get_recent_events(camera_id, limit=500)
    recent = [e for e in events if e["ts"] >= cutoff]
    profiles = _profiles_seen_since(cutoff)
    n_reg = sum(1 for p in profiles if p["is_regular"])
    n_unk = sum(1 for p in profiles if not p["is_regular"])
    vel   = trend_analyzer.velocity(camera_id)
    anom  = trend_analyzer._detect_anomalies(event_db.get_hourly_heatmap(camera_id, 5))

    context = (
        f"Events last 24h: {len(recent)}\n"
        f"Persons: {len(profiles)} ({n_reg} regulars, {n_unk} unknowns)\n"
        f"Velocity trend: {vel['trend']} ({vel['delta_pct']:+.1f}%)\n"
        f"Anomalies: {len(anom)}\n"
        f"Profiles: {chr(10).join(p['label'] for p in profiles[:5])}"
    )
    llm_resp = _llm_query(text, context)
    if llm_resp:
        return {"intent": "summary", "answer": llm_resp, "data": {"events": len(recent), "profiles": profiles}}

    lines = [f"▸ {len(recent)} events in last 24h across {len(set(e['camera_id'] for e in recent))} camera(s)."]
    if profiles:
        lines.append(f"▸ {len(profiles)} person(s) detected: {n_reg} regular(s), {n_unk} unknown(s).")
    if vel["trend"] != "stable":
        arrow = "▲" if vel["trend"] == "rising" else "▼"
        lines.append(f"▸ Activity trend {arrow} {vel['delta_pct']:+.1f}% vs last week.")
    if anom:
        lines.append(f"▸ {len(anom)} statistical anomaly spike(s) detected in baseline.")
    if not recent:
        lines = ["▸ No activity in the last 24 hours. All quiet."]
    return {"intent": "summary", "answer": "\n".join(lines), "data": {"events": len(recent), "profiles": profiles}}


def _handle_who_today(text: str, camera_id: Optional[str]) -> dict:
    ts_from, ts_to = _parse_time_range(text)
    profiles = _profiles_seen_since(ts_from)
    if not profiles:
        answer = "▸ No persons logged in that period."
    else:
        answer = f"▸ {len(profiles)} person(s) detected:\n" + _fmt_profiles(profiles)
    ctx = f"Persons in window: {[p['label'] for p in profiles]}"
    llm = _llm_query(text, ctx)
    return {"intent": "who_today", "answer": llm or answer, "data": {"profiles": profiles}}


def _handle_stranger(text: str, camera_id: Optional[str]) -> dict:
    cutoff = (datetime.now(tz=timezone.utc) - timedelta(hours=48)).timestamp()
    profiles = _profiles_seen_since(cutoff)
    strangers = [p for p in profiles if not p["is_regular"]]
    if not strangers:
        answer = "▸ No unknown individuals detected in last 48h. All visitors are known regulars."
    else:
        lines = [f"▸ {len(strangers)} unknown individual(s) detected:"]
        for s in strangers:
            cams = ", ".join(c.upper() for c in s.get("cameras", []))
            lines.append(f"  ◉ {s['label']} · last seen {_fmtts(s['last_seen'])} on {cams}")
        answer = "\n".join(lines)
    ctx = f"Unknown persons: {[s['label'] for s in strangers]}"
    llm = _llm_query(text, ctx)
    return {"intent": "stranger_check", "answer": llm or answer, "data": {"strangers": strangers}}


def _handle_anomaly(text: str, camera_id: Optional[str]) -> dict:
    heatmap = event_db.get_hourly_heatmap(camera_id, 5)
    anomalies = trend_analyzer._detect_anomalies(heatmap)
    if not anomalies:
        answer = "▸ No statistical anomalies in last 5 weeks. Baseline activity is nominal."
    else:
        lines = [f"▸ {len(anomalies)} anomaly spike(s) detected:"]
        for a in anomalies[:5]:
            lines.append(f"  ⚡ {a['day']} {a['label']} — {a['count']} events (z={a['z_score']})")
        answer = "\n".join(lines)
    ctx = f"Anomalies: {anomalies[:3]}"
    llm = _llm_query(text, ctx)
    return {"intent": "anomaly_check", "answer": llm or answer, "data": {"anomalies": anomalies}}


def _handle_velocity(text: str, camera_id: Optional[str]) -> dict:
    vel = trend_analyzer.velocity(camera_id)
    arrow = "▲" if vel["trend"] == "rising" else ("▼" if vel["trend"] == "falling" else "─")
    answer = (
        f"▸ Activity trend: {arrow} {vel['trend'].upper()} ({vel['delta_pct']:+.1f}%)\n"
        f"▸ This week: {vel['this_week_count']} events ({vel['this_week_daily']}/day)\n"
        f"▸ Last week: {vel['last_week_count']} events ({vel['last_week_daily']}/day)"
    )
    ctx = f"Velocity: this_week={vel['this_week_count']}, last_week={vel['last_week_count']}, trend={vel['trend']}"
    llm = _llm_query(text, ctx)
    return {"intent": "velocity", "answer": llm or answer, "data": vel}


def _handle_camera_compare(text: str, camera_id: Optional[str]) -> dict:
    comp = trend_analyzer.camera_comparison()
    if not comp:
        answer = "▸ No camera data yet."
    else:
        lines = ["▸ Camera activity (7 days):"]
        for c in comp:
            top = f" · top: {c['top_class'].upper()}" if c.get("top_class") else ""
            lines.append(f"  {c['camera_id'].upper():<16} {c['events_7d']} events{top}")
        answer = "\n".join(lines)
    ctx = str(comp)
    llm = _llm_query(text, ctx)
    return {"intent": "camera_compare", "answer": llm or answer, "data": {"comparison": comp}}


def _handle_count(text: str, camera_id: Optional[str]) -> dict:
    ts_from, ts_to = _parse_time_range(text)
    events = event_db.get_events_in_range(ts_from, ts_to, camera_id)
    profiles = _profiles_seen_since(ts_from)
    window = "last 24h" if "today" in text.lower() else "that period"
    answer = (
        f"▸ {len(events)} event(s) in {window}.\n"
        f"▸ {len(profiles)} unique person profile(s) active."
    )
    ctx = f"Events: {len(events)}, persons: {len(profiles)}"
    llm = _llm_query(text, ctx)
    return {"intent": "count_query", "answer": llm or answer, "data": {"event_count": len(events), "person_count": len(profiles)}}


def _handle_time_query(text: str, camera_id: Optional[str]) -> dict:
    ts_from, ts_to = _parse_time_range(text)
    events = event_db.get_events_in_range(ts_from, ts_to, camera_id)
    dt_from = datetime.fromtimestamp(ts_from, tz=timezone.utc)
    dt_to   = datetime.fromtimestamp(ts_to, tz=timezone.utc)
    window  = f"{dt_from.strftime('%H:%M')}–{dt_to.strftime('%H:%M')}"
    if not events:
        answer = f"▸ No events recorded between {window}."
    else:
        # Count detections
        dets = event_db.get_detection_summary(camera_id, weeks=0)
        answer = f"▸ {len(events)} event(s) between {window}.\n"
        cams = list({e["camera_id"] for e in events})
        answer += f"▸ Cameras: {', '.join(c.upper() for c in cams)}"
        if dets:
            answer += f"\n▸ Top detection: {dets[0]['class_name'].upper()} ({dets[0]['count']}×)"
    ctx = f"Events in window {window}: {len(events)}"
    llm = _llm_query(text, ctx)
    return {"intent": "time_query", "answer": llm or answer, "data": {"events": len(events)}}


def _handle_recent(text: str, camera_id: Optional[str]) -> dict:
    limit = 10
    m = re.search(r'last (\d+)', text.lower())
    if m:
        limit = int(m.group(1))
    events = event_db.get_recent_events(camera_id, limit)
    if not events:
        answer = "▸ No events on record yet."
    else:
        lines = [f"▸ {len(events)} most recent event(s):"]
        for ev in events[:8]:
            dt   = datetime.fromtimestamp(ev["ts"], tz=timezone.utc)
            tags = (ev.get("tags") or "").replace(",", " ·")
            lines.append(f"  {dt.strftime('%a %H:%M')} [{ev['camera_id'].upper()}] {tags or '—'}")
        answer = "\n".join(lines)
    ctx = f"Recent events: {[{'ts': e['ts'], 'cam': e['camera_id'], 'tags': e.get('tags','')} for e in events[:5]]}"
    llm = _llm_query(text, ctx)
    return {"intent": "recent_events", "answer": llm or answer, "data": {"events": events[:10]}}


def _handle_person_info(text: str, camera_id: Optional[str]) -> dict:
    # Try to extract a profile ID from the text
    m = re.search(r'(?:profile[-\s]?|regular[-\s]?|unknown[-\s]?)(\d+)', text.lower())
    if m:
        pid = int(m.group(1))
        p   = event_db.get_profile(pid)
        if not p:
            return {"intent": "person_info", "answer": f"▸ Profile {pid} not found.", "data": {}}
        sightings = event_db.get_profile_sightings(pid)
        import json as _json
        label = p.get("label") or (f"REGULAR-{pid:03d}" if p["sightings"] >= 4 else f"UNKNOWN-{pid:03d}")
        cameras = _json.loads(p["cameras"])
        first = _fmtts(p["first_seen"])
        last  = _fmtts(p["last_seen"])
        recent_s = sightings[:5]
        ctx = f"Profile {pid}: label={label}, sightings={p['sightings']}, cameras={cameras}, first={first}, last={last}"
        llm = _llm_query(text, ctx)
        if llm:
            return {"intent": "person_info", "answer": llm, "data": {"profile": dict(p)}}
        lines = [
            f"▸ {label}",
            f"▸ Sightings: {p['sightings']}×",
            f"▸ Cameras: {', '.join(c.upper() for c in cameras)}",
            f"▸ First seen: {first}",
            f"▸ Last seen: {last}",
        ]
        if recent_s:
            lines.append("▸ Recent:")
            for s in recent_s[:3]:
                lines.append(f"  {_fmtts(s['ts'])} · {s['cam_id'].upper()}")
        return {"intent": "person_info", "answer": "\n".join(lines), "data": {"profile": dict(p)}}
    # No specific profile — list all
    profiles = event_db.get_all_profiles()[:6]
    lines = [f"▸ {len(profiles)} profile(s) on record:"]
    import json as _json
    for p in profiles:
        label = p.get("label") or (f"REGULAR-{p['id']:03d}" if p["sightings"] >= 4 else f"UNKNOWN-{p['id']:03d}")
        lines.append(f"  [{p['id']:03d}] {label} · {p['sightings']}× sightings")
    return {"intent": "person_info", "answer": "\n".join(lines), "data": {"profiles": profiles}}


def _handle_watchlist_add(text: str, camera_id: Optional[str]) -> dict:
    rule = add_watchlist_rule(text)
    answer = (
        f"▸ WATCHLIST RULE #{rule['id']} ACTIVATED\n"
        f"▸ Condition: {rule['condition'].upper().replace('_', ' ')}\n"
        f"▸ Raw directive: \"{rule['raw']}\"\n"
        f"▸ System is now monitoring. Alerts will appear in the ALERTS panel."
    )
    return {"intent": "watchlist_add", "answer": answer, "data": {"rule": rule}}


def _handle_watchlist_show(text: str, camera_id: Optional[str]) -> dict:
    rules = get_watchlist()
    if not rules:
        answer = "▸ No active watchlist rules. Use 'alert me when...' to add one."
    else:
        lines = [f"▸ {len(rules)} active watchlist rule(s):"]
        for r in rules:
            lines.append(f"  #{r['id']} [{r['condition'].upper()}] {r['raw']}")
        answer = "\n".join(lines)
    return {"intent": "watchlist_show", "answer": answer, "data": {"rules": rules}}


def _handle_earthquake(text: str, camera_id: Optional[str]) -> dict:
    feeds = _get_feeds()
    if not feeds:
        return {"intent": "earthquake", "answer": "▸ Intel feeds module unavailable.", "data": {}}
    items = feeds.get_earthquakes()
    if not items:
        return {"intent": "earthquake",
                "answer": "▸ No earthquakes detected in the last data pull.\n▸ USGS monitoring active for 200km radius.",
                "data": {"count": 0}}
    lines = [f"▸ SEISMIC REPORT — {len(items)} earthquake(s) within monitoring radius:"]
    for q in items[:8]:
        mag    = q["raw"].get("mag", "?")
        depth  = q["raw"].get("depth", "?")
        dist   = f" · {q['distance_km']:.0f}km from home" if q.get("distance_km") else ""
        age_h  = (time.time() - q["ts"]) / 3600
        age_s  = f"{age_h:.1f}h ago" if age_h < 24 else f"{age_h/24:.1f}d ago"
        sev    = q["severity"]
        lines.append(f"  [{sev}] M{mag} — {q['location']}")
        lines.append(f"    Depth {depth:.0f}km{dist} · {age_s}")
    big = [q for q in items if (q["raw"].get("mag") or 0) >= 3.0]
    if big:
        lines.append(f"\n▸ ⚠ {len(big)} event(s) M3.0+ detected. Monitor aftershock sequence.")
    else:
        lines.append("\n▸ No significant events (M3.0+). Microseismic activity nominal.")
    ctx = f"Earthquakes:\n" + "\n".join(f"M{q['raw'].get('mag',0)} {q['location']}" for q in items[:5])
    llm = _llm_query(text, ctx)
    return {"intent": "earthquake", "answer": llm or "\n".join(lines), "data": {"items": items}}


def _handle_weather_alert(text: str, camera_id: Optional[str]) -> dict:
    feeds = _get_feeds()
    if not feeds:
        return {"intent": "weather_alert", "answer": "▸ Intel feeds module unavailable.", "data": {}}
    items = feeds.get_weather_alerts()
    if not items:
        return {"intent": "weather_alert",
                "answer": "▸ No active NWS alerts for your area.\n▸ Atmospheric conditions: nominal.",
                "data": {"count": 0}}
    lines = [f"▸ NWS ACTIVE ALERTS — {len(items)} alert(s) for {intel_feeds_loc()}:"]
    for a in items:
        lines.append(f"\n  [{a['severity']}] {a['title']}")
        lines.append(f"    {a['detail'][:200]}")
        lines.append(f"    Area: {a['location'][:80]}")
    ctx = "\n".join(f"[{a['severity']}] {a['title']}: {a['detail'][:100]}" for a in items)
    llm = _llm_query(text, ctx)
    return {"intent": "weather_alert", "answer": llm or "\n".join(lines), "data": {"items": items}}


def _handle_fire_intel(text: str, camera_id: Optional[str]) -> dict:
    feeds = _get_feeds()
    if not feeds:
        return {"intent": "fire_intel", "answer": "▸ Intel feeds module unavailable.", "data": {}}
    items = feeds.get_fire_incidents()
    if not items:
        return {"intent": "fire_intel",
                "answer": "▸ No active CAL FIRE incidents in SoCal region.\n▸ Fire threat: NOMINAL.",
                "data": {"count": 0}}
    lines = [f"▸ CAL FIRE ACTIVE — {len(items)} incident(s) in region:"]
    for f in items[:6]:
        dist  = f" · {f['distance_km']:.0f}km from home" if f.get("distance_km") else ""
        lines.append(f"\n  [{f['severity']}] {f['title']}")
        lines.append(f"    {f['detail'][:150]}{dist}")
    red = [f for f in items if f["severity"] == "RED"]
    if red:
        lines.append(f"\n▸ ⚠ {len(red)} CRITICAL fire(s) — review evacuation routes.")
    ctx = "\n".join(f"[{f['severity']}] {f['title']}: {f['detail'][:100]}" for f in items)
    llm = _llm_query(text, ctx)
    return {"intent": "fire_intel", "answer": llm or "\n".join(lines), "data": {"items": items}}


def _handle_local_incidents(text: str, camera_id: Optional[str]) -> dict:
    feeds = _get_feeds()
    if not feeds:
        return {"intent": "local_incidents", "answer": "▸ Intel feeds module unavailable.", "data": {}}
    items = feeds.get_citizen_incidents()
    if not items:
        return {"intent": "local_incidents",
                "answer": "▸ No active Citizen incidents in your area.\n▸ Local conditions appear nominal.",
                "data": {"count": 0}}
    red_orange = [i for i in items if i["severity"] in ("RED", "ORANGE")]
    lines = [f"▸ LOCAL INCIDENTS (Citizen/911) — {len(items)} active · {len(red_orange)} high-priority:"]
    for inc in items[:8]:
        dist  = f"{inc['distance_km']:.1f}km · " if inc.get("distance_km") is not None else ""
        age_m = int((time.time() - inc["ts"]) / 60) if inc.get("ts") else 0
        age_s = f"{age_m}m ago" if age_m < 60 else f"{age_m//60}h ago"
        lines.append(f"\n  [{inc['severity']}] {inc['category']} — {inc['title']}")
        lines.append(f"    {dist}{inc['location']} · {age_s}")
        if inc.get("detail") and inc["detail"] != inc["title"]:
            lines.append(f"    {inc['detail'][:120]}")
    ctx = "\n".join(f"[{i['severity']}] {i['category']} {i['title']} @ {i['location']}" for i in items[:6])
    llm = _llm_query(text, ctx)
    return {"intent": "local_incidents", "answer": llm or "\n".join(lines), "data": {"items": items}}


def _handle_area_threat(text: str, camera_id: Optional[str]) -> dict:
    feeds = _get_feeds()
    if not feeds:
        return {"intent": "area_threat", "answer": "▸ Intel feeds module unavailable.", "data": {}}
    data    = feeds.get_all_feeds()
    briefing = feeds.generate_briefing()
    llm = _llm_query(text, briefing[:800])
    return {"intent": "area_threat", "answer": llm or briefing, "data": data}


def _handle_plates(text: str, camera_id: Optional[str]) -> dict:
    try:
        import lpr_engine
        plates = lpr_engine.get_plate_log(camera_id, limit=30)
        unique = lpr_engine.get_unique_plates(24)
    except Exception:
        return {"intent": "plates", "answer": "▸ LPR engine unavailable.", "data": {}}
    if not plates:
        return {"intent": "plates",
                "answer": "▸ No license plates logged yet.\n▸ LPR will activate automatically when vehicles are detected on camera.",
                "data": {"count": 0}}
    lines = [f"▸ LICENSE PLATE LOG — {len(unique)} unique plate(s) in last 24h:"]
    flagged = [p for p in unique if p.get("flagged")]
    if flagged:
        lines.append(f"▸ ⚠ {len(flagged)} FLAGGED PLATE(S) DETECTED:")
        for p in flagged:
            lines.append(f"  🚨 {p['plate']} ({p['vehicle_class']}) · {p['camera']} · conf {p['confidence']:.0%}")
    for p in unique[:10]:
        label   = f" [{p['label']}]" if p.get("label") else ""
        flagged_tag = " ⚠FLAGGED" if p.get("flagged") else ""
        age_m   = int((time.time() - p["ts"]) / 60) if p.get("ts") else 0
        age_s   = f"{age_m}m ago" if age_m < 60 else f"{age_m//60}h ago"
        lines.append(f"  {p['plate']}{label}{flagged_tag} · {p['vehicle_class']} · {p['camera']} · {age_s}")
    return {"intent": "plates", "answer": "\n".join(lines), "data": {"plates": unique}}


def intel_feeds_loc() -> str:
    try:
        import intel_feeds
        return intel_feeds.HOME_NAME
    except Exception:
        return "your area"


def _handle_help(text: str, camera_id: Optional[str]) -> dict:
    answer = """▸ PALANTIR — PALM COMMAND AI AGENT

SURVEILLANCE QUERIES:
  "Give me a briefing / what happened today"
  "Who was here last night / this morning"
  "Any strangers in the last 48 hours"
  "What happened at 2am"
  "How many events this week"
  "Any unusual activity / anomalies"
  "Which camera is busiest"
  "Is activity increasing / what's the trend"

PERSON INTELLIGENCE:
  "Tell me about REGULAR-001"
  "Who is profile 3"
  "Show me recent events"
  "Last 20 events"

WATCHLIST / RULES:
  "Alert me when an unknown person appears"
  "Watch for vehicles after 10pm"
  "Notify me if 3 or more people"
  "Show my watchlist"

EXTERNAL INTELLIGENCE (live data):
  "Any earthquakes nearby"
  "Seismic activity report"
  "Any weather alerts / red flag warning"
  "Active wildfires near me"
  "Local crime / what's happening in the area"
  "Area threat level / how safe is it"
  "License plates seen today"

MISSION INTEL:
  Powered by PALM COMMAND database + NWS · USGS · CAL FIRE · Citizen
  Kalman filter tracking · LPR active on vehicle detections
  Set OPENAI_API_KEY or ANTHROPIC_API_KEY for LLM upgrade"""
    return {"intent": "help", "answer": answer, "data": {}}


def _handle_general(text: str, camera_id: Optional[str]) -> dict:
    # Try LLM with current status as context
    events = event_db.get_recent_events(camera_id, 20)
    profiles = event_db.get_all_profiles()[:5]
    vel = trend_analyzer.velocity(camera_id)
    import json as _json
    ctx = (
        f"Recent events: {len(events)}\n"
        f"Known profiles: {len(profiles)}\n"
        f"Activity trend: {vel['trend']} ({vel['delta_pct']:+.1f}%)\n"
        f"Recent cams: {list({e['camera_id'] for e in events[:5]})}"
    )
    llm = _llm_query(text, ctx)
    if llm:
        return {"intent": "general", "answer": llm, "data": {}}
    return {
        "intent":  "general",
        "answer":  "▸ Command not recognised. Type 'help' for available queries.\n▸ Tip: set OPENAI_API_KEY for natural language understanding.",
        "data":    {}
    }


# ── Main query entry point ────────────────────────────────────────

def query(
    text: str,
    camera_id: Optional[str] = None,
) -> dict:
    """
    Process a natural language query and return a structured response.

    Returns:
      {
        "intent":    str,          # classified intent
        "answer":    str,          # plain-text response for display
        "data":      dict,         # structured data (optional)
        "ts":        float,        # response timestamp
        "llm_used":  bool,         # True if LLM was used
        "has_llm":   bool,         # True if LLM is available
      }
    """
    text      = text.strip()
    intent    = _classify_intent(text)
    llm_avail = _init_llm()

    handlers = {
        "summary":        _handle_summary,
        "who_today":      _handle_who_today,
        "stranger_check": _handle_stranger,
        "anomaly_check":  _handle_anomaly,
        "velocity":       _handle_velocity,
        "camera_compare": _handle_camera_compare,
        "count_query":    _handle_count,
        "time_query":     _handle_time_query,
        "recent_events":  _handle_recent,
        "person_info":    _handle_person_info,
        "watchlist_add":  _handle_watchlist_add,
        "watchlist_show": _handle_watchlist_show,
        # External intelligence feeds
        "earthquake":      _handle_earthquake,
        "weather_alert":   _handle_weather_alert,
        "fire_intel":      _handle_fire_intel,
        "local_incidents": _handle_local_incidents,
        "area_threat":     _handle_area_threat,
        "plates":          _handle_plates,
        "help":           _handle_help,
        "general":        _handle_general,
    }

    try:
        result = handlers.get(intent, _handle_general)(text, camera_id)
    except Exception as e:
        print(f"[agent] query error: {e}", flush=True)
        result = {"intent": intent, "answer": f"▸ Query error: {e}", "data": {}}

    result["ts"]       = time.time()
    result["llm_used"] = llm_avail and result.get("intent") not in ("help",)
    result["has_llm"]  = llm_avail

    return result
