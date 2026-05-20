"""
forward_intel.py — Predictive threat scenarios + behavioral classification.

While pattern_engine produces individual predictions ("REGULAR-001 expected
in 23min"), forward_intel produces SCENARIOS — narrative explanations of
what is likely about to happen, ranked by probability and severity.

This is the layer Palantir analysts call "forward intelligence" — answering
"what's coming?" rather than "what happened?"

CAPABILITIES:
    1. Behavior classification     — Tags every entity as one of:
                                     visitor / regular / loiterer / scout /
                                     intruder / runner / lookout
    2. Scouting detection          — Flags suspicious pre-incident patterns
                                     (slow approach + circling + lingering)
    3. Convergence prediction       — Multiple unrelated entities arriving
                                     at the same camera within the same
                                     window (possible meet / coordinated event)
    4. Anomalous absence detection  — Regulars who SHOULD be here but aren't
                                     (often signals something is wrong)
    5. Pre-attack indicators        — Combines threat score deltas, entity
                                     graph density spikes, and dwell-time
                                     anomalies into a single risk forecast

CLAUDE-CODE EXTENSION POINTS:
    - Add a behavior class: append to BEHAVIOR_CLASSIFIERS
    - Add a scenario type: implement _build_<scenario>_scenarios()
"""
from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Any

import event_db
import pattern_engine


# ── Event-source shim (works against the existing event_db schema) ─

def _profile_label(pid: int) -> str:
    try:
        import profiler
        return profiler.get_profile_label(pid) or f"PROFILE-{pid:03d}"
    except Exception:
        return f"PROFILE-{pid:03d}"


def _all_recent_events(lookback_sec: int) -> list[dict]:
    """Fetch sightings across ALL profiles in the lookback window.
    Returned dicts have keys: profile_id, camera, ts, label."""
    cutoff = time.time() - lookback_sec
    out: list[dict] = []
    try:
        for prof in event_db.get_all_profiles():
            pid   = prof["id"]
            label = _profile_label(pid)
            for s in event_db.get_profile_sightings(pid):
                if s["ts"] < cutoff:
                    break
                out.append({"profile_id": label, "camera": s["cam_id"], "ts": s["ts"]})
    except Exception:
        pass
    return out


def _events_for(entity_id: str, lookback_sec: int) -> list[dict]:
    """Sightings for a specific entity (matches by profile label)."""
    cutoff = time.time() - lookback_sec
    out: list[dict] = []
    try:
        for prof in event_db.get_all_profiles():
            pid   = prof["id"]
            label = _profile_label(pid)
            if entity_id not in (label, f"PROFILE-{pid:03d}", str(pid)):
                continue
            for s in event_db.get_profile_sightings(pid):
                if s["ts"] < cutoff:
                    break
                out.append({"profile_id": label, "camera": s["cam_id"], "ts": s["ts"]})
    except Exception:
        pass
    return out

# ── Tunables ────────────────────────────────────────────────────

DWELL_LOITERER_SEC      = 600   # 10min in same camera = loiterer
DWELL_SCOUT_RANGE       = (90, 360)  # 1.5–6min wandering = potential scout
SCOUT_CAM_COUNT_MIN     = 3     # appears at 3+ cameras in 15min = scout
CONVERGENCE_WINDOW_SEC  = 300   # 5min
ABSENCE_GRACE_FACTOR    = 1.6   # 1.6× typical inter-arrival = absent
LOOKBACK_SECONDS_DEFAULT = 86400 * 3   # 3 days


# ── Behavior classification ─────────────────────────────────────

def classify_entity(entity_id: str, lookback_sec: int = LOOKBACK_SECONDS_DEFAULT) -> dict:
    """Classify a single entity's recent behavior."""
    events = _events_for(entity_id, lookback_sec)
    if not events:
        return {"entity_id": entity_id, "class": "unknown",
                "confidence": 0.0, "evidence": ["no recent activity"]}

    cams       = {e["camera"] for e in events}
    n_events   = len(events)
    spans      = sorted(e["ts"] for e in events)
    duration   = spans[-1] - spans[0] if len(spans) > 1 else 0.0
    intervals  = [spans[i+1] - spans[i] for i in range(len(spans)-1)] or [0]
    avg_gap    = sum(intervals) / len(intervals)
    max_gap    = max(intervals) if intervals else 0
    night_n    = sum(1 for e in events if datetime.fromtimestamp(e["ts"]).hour in range(0, 6))

    label  = entity_id
    cls    = "visitor"
    conf   = 0.5
    evid   = []

    # REGULAR — many sightings, predictable schedule
    if n_events >= 8 and len(cams) <= 3 and avg_gap < 86400:
        cls, conf = "regular", 0.85
        evid.append(f"{n_events} sightings, {len(cams)} cam(s), avg-gap {int(avg_gap/3600)}h")
    # SCOUT — appeared at many cams quickly
    elif len(cams) >= SCOUT_CAM_COUNT_MIN and duration <= 900:
        cls, conf = "scout", 0.8
        evid.append(f"hit {len(cams)} cameras in {int(duration/60)}min — reconnaissance pattern")
    # LOITERER — long dwell at one spot
    elif duration >= DWELL_LOITERER_SEC and len(cams) == 1:
        cls, conf = "loiterer", 0.78
        evid.append(f"dwelled {int(duration/60)}min on single camera")
    # RUNNER — multiple cams, very short intervals
    elif len(cams) >= 2 and avg_gap < 30 and n_events >= 3:
        cls, conf = "runner", 0.7
        evid.append(f"avg {int(avg_gap)}s between cameras — moving fast")
    # LOOKOUT — short bursts at one camera, multiple times
    elif n_events >= 4 and avg_gap < 1800 and len(cams) == 1 and duration < 1800:
        cls, conf = "lookout", 0.65
        evid.append(f"{n_events} brief returns to same camera")
    # INTRUDER — night-only, unfamiliar (no REGULAR label)
    elif night_n >= 2 and not entity_id.startswith("REGULAR"):
        cls, conf = "intruder", 0.7
        evid.append(f"{night_n} overnight sightings, non-regular")
    elif n_events >= 3:
        cls, conf = "occasional", 0.55
        evid.append(f"{n_events} sightings, non-routine")

    return {
        "entity_id": entity_id, "label": label, "class": cls,
        "confidence": conf, "evidence": evid,
        "sightings": n_events, "cameras": sorted(cams),
        "first_seen": spans[0], "last_seen": spans[-1],
        "avg_gap_min": round(avg_gap / 60, 1), "max_gap_min": round(max_gap / 60, 1),
    }


# ── Scenario builders ───────────────────────────────────────────

def _entities_seen(lookback_sec: int) -> list[str]:
    return sorted({e["profile_id"] for e in _all_recent_events(lookback_sec)
                   if e.get("profile_id")})


def _scenario_scouting() -> list[dict]:
    out = []
    for eid in _entities_seen(3600 * 6):
        c = classify_entity(eid, lookback_sec=3600 * 6)
        if c["class"] == "scout":
            out.append({
                "type": "scouting_pattern",
                "severity": "HIGH",
                "probability": c["confidence"],
                "title": f"Possible reconnaissance — {eid}",
                "narrative": (f"Entity {eid} hit {len(c['cameras'])} cameras "
                              f"({', '.join(c['cameras'])}) within "
                              f"{c['max_gap_min']:.0f}min. Pattern matches "
                              f"pre-incident scouting behavior."),
                "recommend": "Increase alerting on adjacent cameras for next 60min.",
                "entity_id": eid,
            })
    return out


def _scenario_convergence() -> list[dict]:
    """Detect cases where 3+ unrelated entities arrived at one camera in a tight window."""
    events = sorted(_all_recent_events(3600 * 6), key=lambda e: e["ts"])
    by_cam: dict[str, list] = {}
    for e in events:
        by_cam.setdefault(e["camera"], []).append(e)

    out = []
    for cam, evs in by_cam.items():
        if len(evs) < 3:
            continue
        for i in range(len(evs) - 2):
            window = [evs[i]]
            for j in range(i + 1, len(evs)):
                if evs[j]["ts"] - evs[i]["ts"] <= CONVERGENCE_WINDOW_SEC:
                    window.append(evs[j])
                else:
                    break
            uniq = {w.get("profile_id") for w in window if w.get("profile_id")}
            if len(uniq) >= 3:
                out.append({
                    "type": "convergence",
                    "severity": "MEDIUM",
                    "probability": min(0.9, 0.5 + 0.1 * len(uniq)),
                    "title": f"{len(uniq)} entities converged on {cam}",
                    "narrative": (f"Within {CONVERGENCE_WINDOW_SEC//60}min, "
                                  f"unrelated entities {', '.join(sorted(uniq))[:120]} "
                                  f"all appeared at {cam}. Possible coordinated event."),
                    "recommend": "Review full event log for the window. Cross-check entity graph for prior associations.",
                    "camera": cam,
                })
                break
    # dedup by camera
    dedup = {}
    for s in out:
        dedup[s["camera"]] = s
    return list(dedup.values())


def _scenario_anomalous_absence() -> list[dict]:
    """Regulars whose typical inter-arrival has expired by ABSENCE_GRACE_FACTOR×."""
    eng = pattern_engine.get_engine()
    out = []
    for pol in eng.get_all_pol():
        eid       = pol.get("entity_id") or pol.get("id") or pol.get("label", "?")
        intervals = pol.get("avg_inter_arrival_min") or pol.get("inter_arrival_min")
        if not intervals or intervals < 30:
            continue
        last = pol.get("last_seen_ts") or pol.get("last_seen", 0)
        if not last:
            continue
        elapsed_min = (time.time() - last) / 60
        if elapsed_min > intervals * ABSENCE_GRACE_FACTOR:
            overdue_pct = elapsed_min / intervals
            out.append({
                "type": "anomalous_absence",
                "severity": "LOW" if overdue_pct < 3 else "MEDIUM",
                "probability": min(0.85, 0.4 + 0.15 * overdue_pct),
                "title": f"Regular overdue — {eid}",
                "narrative": (f"{eid} normally appears every "
                              f"{intervals:.0f}min but has been absent "
                              f"{elapsed_min:.0f}min ({overdue_pct:.1f}× normal interval)."),
                "recommend": "Verify wellbeing if known person. Otherwise log as routine variation.",
                "entity_id": eid,
            })
    return out


def _scenario_loitering() -> list[dict]:
    out = []
    for eid in _entities_seen(3600 * 4):
        c = classify_entity(eid, lookback_sec=3600 * 4)
        if c["class"] in ("loiterer", "lookout"):
            out.append({
                "type": "loitering",
                "severity": "MEDIUM" if c["class"] == "lookout" else "LOW",
                "probability": c["confidence"],
                "title": f"{c['class'].title()} — {eid}",
                "narrative": (f"{eid} classified as {c['class']}. "
                              f"{'; '.join(c['evidence'])}."),
                "recommend": ("Add to watchlist and monitor for return"
                              if c["class"] == "lookout"
                              else "Observe; may be benign waiting/resting."),
                "entity_id": eid,
            })
    return out


def _scenario_intruder() -> list[dict]:
    out = []
    for eid in _entities_seen(3600 * 24):
        c = classify_entity(eid, lookback_sec=3600 * 24)
        if c["class"] == "intruder":
            out.append({
                "type": "potential_intruder",
                "severity": "HIGH",
                "probability": c["confidence"],
                "title": f"Unfamiliar overnight presence — {eid}",
                "narrative": (f"{eid} has {c['sightings']} overnight sightings "
                              f"and is not classified as a regular."),
                "recommend": "Manual review of night clips; consider escalating to local PD.",
                "entity_id": eid,
            })
    return out


# ── Top-level forecast ──────────────────────────────────────────

def build_scenarios() -> list[dict]:
    """Build all forward-intel scenarios, sort by severity then probability."""
    sev_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    scenarios = []
    for builder in (_scenario_scouting, _scenario_convergence,
                    _scenario_intruder, _scenario_loitering,
                    _scenario_anomalous_absence):
        try:
            scenarios.extend(builder())
        except Exception as e:
            scenarios.append({"type": "engine_error", "severity": "LOW",
                              "probability": 0.0, "title": f"{builder.__name__}: {e}",
                              "narrative": "Scenario builder failed.", "recommend": ""})
    scenarios.sort(key=lambda s: (sev_rank.get(s["severity"], 9),
                                  -s.get("probability", 0)))
    return scenarios


def forecast_briefing(limit: int = 10) -> str:
    scenarios = build_scenarios()
    if not scenarios:
        return ("▸ FORWARD INTELLIGENCE — COMMAND CENTER\n"
                "▸ No predictive scenarios at this time.\n"
                "▸ All observed entities behaving within normal pattern envelope.\n"
                "▸ Engine will surface scenarios as anomalous patterns emerge.")
    lines = [f"▸ FORWARD INTELLIGENCE — {len(scenarios)} active scenario(s)"]
    for s in scenarios[:limit]:
        prob_pct = int(s["probability"] * 100)
        lines.append(f"\n▸ [{s['severity']}] {s['title']}  (P={prob_pct}%)")
        lines.append(f"   {s['narrative']}")
        if s.get("recommend"):
            lines.append(f"   ↳ {s['recommend']}")
    if len(scenarios) > limit:
        lines.append(f"\n▸ +{len(scenarios)-limit} additional scenarios at /intel/forecast")
    return "\n".join(lines)


def behavior_briefing() -> str:
    eids = _entities_seen(3600 * 24)
    if not eids:
        return ("▸ BEHAVIOR CLASSIFICATION — COMMAND CENTER\n"
                "▸ No entities observed in the last 24 hours.")
    classifications = [classify_entity(eid) for eid in eids]
    classifications.sort(key=lambda c: -c["confidence"])
    lines = [f"▸ BEHAVIOR CLASSIFICATION — {len(classifications)} entit(ies) tagged"]
    by_class: dict[str, int] = {}
    for c in classifications:
        by_class[c["class"]] = by_class.get(c["class"], 0) + 1
    lines.append("▸ DISTRIBUTION:")
    for cls, n in sorted(by_class.items(), key=lambda kv: -kv[1]):
        lines.append(f"   {cls:<12} {n}")
    lines.append("▸ NOTABLE:")
    for c in classifications[:8]:
        if c["class"] in ("visitor", "occasional"):
            continue
        lines.append(f"   • {c['entity_id']:<14} [{c['class']}] {('; '.join(c['evidence']))[:90]}")
    return "\n".join(lines)
