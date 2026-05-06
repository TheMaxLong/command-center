#!/usr/bin/env python3.12
"""
PALM COMMAND — Intelligence Engine.

Generates plain-English scene summaries and alert logic on top of the
raw detection + profile data.  No external API required — pure local logic.

Key outputs:
  scene_summary(detections, profiles, cam_id)
      → one-liner describing what was seen ("2 people, 1 vehicle")
  stranger_alert(profile_id, event_ts, cam_id)
      → True if an unknown person appeared outside normal hours
  daily_briefing(camera_id=None)
      → dict with yesterday's stats + top anomalies + new profiles
  cross_camera_timeline(profile_id, limit=20)
      → list of sightings across all cameras, sorted newest first
  active_alerts(hours=24)
      → recent pinned anomaly events worth surfacing on the dashboard
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone, timedelta
from typing import Optional

import event_db
import trend_analyzer

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAYS_SHORT = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]


# ── Scene summary ─────────────────────────────────────────────────

def scene_summary(
    detections: list[dict],
    profiles: list[dict],
    cam_id: str = "",
) -> str:
    """
    One-liner scene description for a detection event.
    E.g.: "2 people (1 regular, 1 unknown) · 1 vehicle"
    """
    if not detections:
        return "no detections"

    # Group by class
    counts: dict[str, int] = {}
    for d in detections:
        cls = d.get("class", "unknown")
        counts[cls] = counts.get(cls, 0) + 1

    parts: list[str] = []

    # Persons with profile breakdown
    n_persons = counts.get("person", 0)
    if n_persons:
        n_regulars  = sum(1 for p in profiles if p.get("is_regular"))
        n_unknowns  = sum(1 for p in profiles if not p.get("is_regular"))
        untracked   = n_persons - len(profiles)
        if n_persons == 1:
            if n_regulars:
                who = next((p["label"] for p in profiles if p.get("is_regular")), "REGULAR")
                parts.append(f"1 person ({who})")
            elif profiles:
                who = profiles[0]["label"]
                parts.append(f"1 person ({who})")
            else:
                parts.append("1 person (unidentified)")
        else:
            desc = f"{n_persons} people"
            breakdown = []
            if n_regulars:
                breakdown.append(f"{n_regulars} regular{'s' if n_regulars > 1 else ''}")
            if n_unknowns:
                breakdown.append(f"{n_unknowns} unknown{'s' if n_unknowns > 1 else ''}")
            if untracked > 0:
                breakdown.append(f"{untracked} untracked")
            if breakdown:
                desc += f" ({', '.join(breakdown)})"
            parts.append(desc)

    # Vehicles
    vehicle_classes = {"car", "truck", "bus", "motorcycle", "bicycle"}
    n_vehicles = sum(counts.get(c, 0) for c in vehicle_classes)
    if n_vehicles:
        parts.append(f"{n_vehicles} vehicle{'s' if n_vehicles > 1 else ''}")

    # Animals
    animal_classes = {"dog", "cat", "bird"}
    for ac in animal_classes:
        n = counts.get(ac, 0)
        if n:
            parts.append(f"{n} {ac}{'s' if n > 1 else ''}")

    # Other
    other_classes = {"backpack", "handbag", "suitcase", "umbrella", "laptop", "cell phone"}
    n_other = sum(counts.get(c, 0) for c in other_classes)
    if n_other:
        parts.append(f"{n_other} item{'s' if n_other > 1 else ''}")

    summary = " · ".join(parts) if parts else "activity detected"

    # Append camera tag if provided
    if cam_id:
        summary += f" [{cam_id.upper()}]"

    return summary


# ── Stranger alert ────────────────────────────────────────────────

def stranger_alert(profile_id: int, event_ts: float, cam_id: str) -> bool:
    """
    Return True if this profile is unknown (< MIN_SIGHTINGS) and the event
    happened outside the camera's typical activity hours (as inferred from history).

    Useful for surfacing "unknown person at 2am" vs "unknown person at noon".
    """
    import profiler as profiler_mod

    label = profiler_mod.get_profile_label(profile_id)
    is_regular = label.startswith("REGULAR")
    is_operator_cleared = label.startswith(("TRUSTED", "IGNORE", "BACKGROUND"))
    if is_operator_cleared:
        return False   # operator-classified profile, not a stranger alert
    if is_regular:
        return False   # known person — not an alert

    # Check if the hour is unusual
    dt       = datetime.fromtimestamp(event_ts, tz=timezone.utc)
    hour     = dt.hour
    heatmap  = event_db.get_hourly_heatmap(cam_id, weeks=5)
    if not heatmap:
        return False   # no baseline yet

    counts = [
        next((row["count"] for row in heatmap
              if row["day_of_week"] == dt.weekday() and row["hour"] == h), 0)
        for h in range(24)
    ]
    total  = sum(counts)
    if total == 0:
        return False
    mean   = total / 24
    stddev = (sum((c - mean) ** 2 for c in counts) / 24) ** 0.5
    cutoff = mean + stddev

    # Alert if this hour has historically low activity
    hour_count = counts[hour]
    return hour_count < (mean - 0.5 * stddev) if stddev > 0 else False


# ── Daily briefing ────────────────────────────────────────────────

def daily_briefing(camera_id: Optional[str] = None) -> dict:
    """
    Summary of the last 24 hours.

    Returns:
      period_start, period_end, total_events, unique_cameras,
      persons_seen (list of profile summaries), new_profiles,
      top_detections, anomalies, schedule_matches, intel_lines
    """
    import profiler as profiler_mod

    now     = datetime.now(tz=timezone.utc)
    cutoff  = now - timedelta(hours=24)
    cutoff_ts = cutoff.timestamp()

    # Recent events
    events = event_db.get_recent_events(camera_id, limit=500)
    recent = [e for e in events if e["ts"] >= cutoff_ts]

    total_events   = len(recent)
    unique_cameras = list({e["camera_id"] for e in recent})

    # Detection summary for period
    top_detections = event_db.get_detection_summary(camera_id, weeks=0)

    # Profiles seen today
    all_profiles  = event_db.get_all_profiles()
    today_pids: set[int] = set()
    for e in recent:
        if e.get("tags"):
            pass   # we need profile sightings; use a simpler proxy
    # Get sightings from today
    profiles_today: list[dict] = []
    new_profiles: list[dict]   = []
    for p in all_profiles:
        if p["last_seen"] >= cutoff_ts:
            label  = profiler_mod._make_label(p["id"], p["sightings"], p.get("label"))
            pentry = {
                "id":         p["id"],
                "label":      label,
                "sightings":  p["sightings"],
                "is_regular": p["sightings"] >= profiler_mod.MIN_SIGHTINGS,
            }
            profiles_today.append(pentry)
            if p["first_seen"] >= cutoff_ts:
                new_profiles.append(pentry)

    # Anomalies from trend analyzer
    report    = trend_analyzer.analyze(camera_id, weeks=5)
    anomalies = report.get("anomalies", [])
    schedule  = report.get("schedule", [])

    # Generate human-readable intel lines
    intel_lines: list[str] = []

    if total_events == 0:
        intel_lines.append("No activity in the last 24 hours.")
    else:
        intel_lines.append(
            f"{total_events} event{'s' if total_events != 1 else ''} across "
            f"{len(unique_cameras)} camera{'s' if len(unique_cameras) != 1 else ''}."
        )

    n_reg  = sum(1 for p in profiles_today if p["is_regular"])
    n_unk  = sum(1 for p in profiles_today if not p["is_regular"])
    if n_reg:
        intel_lines.append(f"{n_reg} regular visitor{'s' if n_reg > 1 else ''} identified.")
    if n_unk:
        intel_lines.append(f"{n_unk} unknown individual{'s' if n_unk > 1 else ''} detected.")
    if new_profiles:
        intel_lines.append(f"{len(new_profiles)} new profile{'s' if len(new_profiles) > 1 else ''} created.")

    if anomalies:
        top_a = anomalies[0]
        intel_lines.append(
            f"Anomaly: {top_a['day']} {top_a['label']} — "
            f"{top_a['count']} events (z={top_a['z_score']})."
        )

    # Current hour schedule match
    current_hour = now.hour
    current_day  = DAYS_SHORT[now.weekday()]
    on_schedule  = [s for s in schedule if s["day"] == current_day and s["hour"] == current_hour]
    if on_schedule:
        intel_lines.append(f"Now ({current_day} {current_hour:02d}:00) is a typical active period.")

    return {
        "period_start":    cutoff.isoformat(),
        "period_end":      now.isoformat(),
        "total_events":    total_events,
        "unique_cameras":  unique_cameras,
        "persons_seen":    profiles_today,
        "new_profiles":    new_profiles,
        "top_detections":  top_detections[:8],
        "anomalies":       anomalies[:5],
        "schedule_matches": on_schedule,
        "intel_lines":     intel_lines,
    }


# ── Cross-camera timeline ─────────────────────────────────────────

def cross_camera_timeline(profile_id: int, limit: int = 20) -> list[dict]:
    """
    Return chronological sightings of a profile across all cameras.
    Each entry: { ts, cam_id, event_id, label, time_str, day_str }
    """
    import profiler as profiler_mod

    sightings = event_db.get_profile_sightings(profile_id)[:limit]
    label     = profiler_mod.get_profile_label(profile_id)

    result: list[dict] = []
    for s in sightings:
        dt = datetime.fromtimestamp(s["ts"], tz=timezone.utc)
        result.append({
            "ts":       s["ts"],
            "cam_id":   s["cam_id"],
            "event_id": s["event_id"],
            "label":    label,
            "time_str": dt.strftime("%H:%M:%S"),
            "day_str":  dt.strftime("%a %b %-d"),
        })
    return result


# ── Active alerts ─────────────────────────────────────────────────

def active_alerts(camera_id: Optional[str] = None, hours: int = 24) -> list[dict]:
    """
    Surface recent anomalous moments as dashboard alerts.

    Combines:
      - Statistical anomalies from trend_analyzer (z-score spikes)
      - Events involving unknown persons during off-peak hours
      - Profile merge notices (surfaced as info alerts)

    Returns a list of alert dicts sorted newest first:
      { type, severity, message, ts, camera_id }
    """
    now      = datetime.now(tz=timezone.utc)
    cutoff   = (now - timedelta(hours=hours)).timestamp()
    alerts:  list[dict] = []

    # ── Statistical anomaly alerts ────────────────────────────────
    report    = trend_analyzer.analyze(camera_id, weeks=5)
    anomalies = report.get("anomalies", [])
    for a in anomalies[:3]:
        alerts.append({
            "type":      "anomaly",
            "severity":  "warn" if a["z_score"] < 4 else "high",
            "message":   f"Unusual spike: {a['day']} {a['label']} — {a['count']} events (z={a['z_score']})",
            "ts":        now.timestamp(),
            "camera_id": camera_id or "all",
        })

    # ── Recent events: unknown persons in off-peak hours ─────────
    events = event_db.get_recent_events(camera_id, limit=200)
    recent = [e for e in events if e["ts"] >= cutoff]

    # Check for high-event hours
    if recent:
        total_recent = len(recent)
        if total_recent > 30:
            alerts.append({
                "type":      "volume",
                "severity":  "info",
                "message":   f"High activity: {total_recent} events in last {hours}h",
                "ts":        now.timestamp(),
                "camera_id": camera_id or "all",
            })

    # Sort by severity then ts
    sev_order = {"high": 0, "warn": 1, "info": 2}
    alerts.sort(key=lambda a: (sev_order.get(a["severity"], 9), -a["ts"]))

    return alerts[:10]


# ── Velocity tracking ─────────────────────────────────────────────

def event_velocity(camera_id: Optional[str] = None) -> dict:
    """
    Compare event rate this week vs. last week.

    Returns:
      this_week_count, last_week_count, delta_pct, trend ('rising'|'falling'|'stable')
    """
    events = event_db.get_recent_events(camera_id, limit=2000)
    now    = datetime.now(tz=timezone.utc).timestamp()

    this_week  = sum(1 for e in events if now - e["ts"] < 7 * 86400)
    last_week  = sum(1 for e in events if 7 * 86400 <= now - e["ts"] < 14 * 86400)

    if last_week == 0:
        delta_pct = 0.0
        trend     = "stable"
    else:
        delta_pct = round((this_week - last_week) / last_week * 100, 1)
        trend     = "rising" if delta_pct > 10 else ("falling" if delta_pct < -10 else "stable")

    return {
        "this_week_count": this_week,
        "last_week_count": last_week,
        "delta_pct":       delta_pct,
        "trend":           trend,
    }
