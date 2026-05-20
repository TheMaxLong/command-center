#!/usr/bin/env python3.12
"""
COMMAND CENTER — Trend analyzer v2.

Adds over v1:
  • velocity() — events-per-day rate this week vs last week
  • pinned_anomalies() — top anomalies with camera breakdown
  • camera_comparison() — side-by-side event counts across cameras
  • analyze() now returns schedule, anomalies, velocity, AND camera_comparison

The /trends endpoint in camera_watcher.py now calls analyze() directly
so ALL of this data reaches the dashboard.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional

import event_db

DAYS       = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
DAYS_LONG  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def analyze(camera_id: Optional[str] = None, weeks: int = 5) -> dict:
    """
    Full trend report ready for JSON serialisation.

    Keys:
      weeks, camera_id, heatmap, schedule, detections, anomalies,
      velocity, camera_comparison, updated_at
    """
    heatmap    = event_db.get_hourly_heatmap(camera_id, weeks)
    detections = event_db.get_detection_summary(camera_id, weeks)
    schedule   = _infer_schedule(heatmap)
    anomalies  = _detect_anomalies(heatmap)
    vel        = velocity(camera_id)
    cam_comp   = camera_comparison() if camera_id is None else []

    return {
        "weeks":              weeks,
        "camera_id":          camera_id,
        "heatmap":            heatmap,
        "schedule":           schedule,
        "detections":         detections,
        "anomalies":          anomalies,
        "velocity":           vel,
        "camera_comparison":  cam_comp,
        "updated_at":         datetime.now(tz=timezone.utc).isoformat(),
    }


# ── Schedule inference ────────────────────────────────────────────

def _infer_schedule(heatmap: list[dict]) -> list[dict]:
    """
    Hours per day whose event count exceeds mean + 1σ for that day.
    Represents recurring patterns (scheduled activity).
    """
    if not heatmap:
        return []

    grid: dict[tuple[int, int], int] = {}
    for row in heatmap:
        grid[(row["day_of_week"], row["hour"])] = row["count"]

    schedule: list[dict] = []
    for day in range(7):
        counts = [grid.get((day, h), 0) for h in range(24)]
        total  = sum(counts)
        if total == 0:
            continue
        mean   = total / 24
        stddev = (sum((c - mean) ** 2 for c in counts) / 24) ** 0.5
        cutoff = mean + stddev

        for hour in range(24):
            c = counts[hour]
            if c > cutoff:
                schedule.append({
                    "day":       DAYS[day],
                    "day_long":  DAYS_LONG[day],
                    "hour":      hour,
                    "label":     f"{hour:02d}:00",
                    "count":     c,
                    "intensity": round((c - mean) / (stddev or 1), 2),
                })

    return sorted(schedule, key=lambda x: (DAYS.index(x["day"]), x["hour"]))


# ── Anomaly detection ─────────────────────────────────────────────

def _detect_anomalies(heatmap: list[dict]) -> list[dict]:
    """
    (day, hour) cells with count > mean + 2σ globally.
    Sorted by z-score descending.
    """
    if len(heatmap) < 3:
        return []

    counts = [row["count"] for row in heatmap]
    mean   = sum(counts) / len(counts)
    var    = sum((c - mean) ** 2 for c in counts) / len(counts)
    stddev = var ** 0.5

    if stddev == 0:
        return []

    anomalies: list[dict] = []
    for row in heatmap:
        z = (row["count"] - mean) / stddev
        if z > 2.0:
            anomalies.append({
                "day":     DAYS[row["day_of_week"]],
                "day_long":DAYS_LONG[row["day_of_week"]],
                "hour":    row["hour"],
                "label":   f"{row['hour']:02d}:00",
                "count":   row["count"],
                "z_score": round(z, 2),
            })

    return sorted(anomalies, key=lambda x: -x["z_score"])


# ── Velocity ──────────────────────────────────────────────────────

def velocity(camera_id: Optional[str] = None) -> dict:
    """
    Events per day: this week vs last week.
    Returns trend direction + daily rates.
    """
    events     = event_db.get_recent_events(camera_id, limit=5000)
    import time
    now        = time.time()

    this_week  = [e for e in events if now - e["ts"] <  7 * 86400]
    last_week  = [e for e in events if 7 * 86400 <= now - e["ts"] < 14 * 86400]

    tw_count = len(this_week)
    lw_count = len(last_week)
    tw_daily = round(tw_count / 7, 1)
    lw_daily = round(lw_count / 7, 1)

    if lw_count == 0:
        delta_pct = 0.0
        trend     = "stable"
    else:
        delta_pct = round((tw_count - lw_count) / lw_count * 100, 1)
        trend = "rising" if delta_pct > 10 else ("falling" if delta_pct < -10 else "stable")

    # Daily breakdown for sparkline (last 14 days)
    daily_counts: dict[str, int] = {}
    for e in events:
        if now - e["ts"] > 14 * 86400:
            continue
        dt  = datetime.fromtimestamp(e["ts"], tz=timezone.utc)
        key = dt.strftime("%a %-d")
        daily_counts[key] = daily_counts.get(key, 0) + 1

    return {
        "this_week_count": tw_count,
        "last_week_count": lw_count,
        "this_week_daily": tw_daily,
        "last_week_daily": lw_daily,
        "delta_pct":       delta_pct,
        "trend":           trend,
        "daily_counts":    daily_counts,
    }


# ── Camera comparison ─────────────────────────────────────────────

def camera_comparison() -> list[dict]:
    """
    Per-camera event count + top detection class for the last 7 days.
    Sorted by event count descending.
    """
    events = event_db.get_recent_events(None, limit=5000)
    import time
    cutoff = time.time() - 7 * 86400

    cam_counts: dict[str, int] = {}
    for e in events:
        if e["ts"] >= cutoff:
            cid = e["camera_id"]
            cam_counts[cid] = cam_counts.get(cid, 0) + 1

    result: list[dict] = []
    for cam_id, count in sorted(cam_counts.items(), key=lambda x: -x[1]):
        top_det = event_db.get_detection_summary(cam_id, weeks=1)
        result.append({
            "camera_id":   cam_id,
            "events_7d":   count,
            "top_class":   top_det[0]["class_name"] if top_det else None,
        })
    return result


# ── Pinned anomalies ──────────────────────────────────────────────

def pinned_anomalies(camera_id: Optional[str] = None, top_n: int = 3) -> list[dict]:
    """Return the top N anomaly cells with camera breakdown."""
    heatmap   = event_db.get_hourly_heatmap(camera_id, weeks=5)
    anomalies = _detect_anomalies(heatmap)
    return anomalies[:top_n]


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json as _json

    ap = argparse.ArgumentParser(description="Print 5-week trend report")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--weeks", type=int, default=5)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    event_db.init_db()
    report = analyze(camera_id=args.camera, weeks=args.weeks)

    if args.json:
        print(_json.dumps(report, indent=2))
    else:
        print(f"\n{'─'*56}")
        print(f"  COMMAND CENTER · Trend Report · Last {args.weeks} weeks")
        if args.camera:
            print(f"  Camera: {args.camera}")
        print(f"{'─'*56}")

        print("\n  TOP DETECTIONS")
        for d in report["detections"][:8]:
            bar = "█" * min(20, d["count"])
            print(f"  {d['class_name']:<14} {d['count']:>4}x  {bar}  ({d['avg_conf']*100:.0f}% avg conf)")

        print("\n  RECURRING SCHEDULE (peaks > mean + 1σ per day)")
        for s in report["schedule"] or [{"day":"—","label":"not enough data yet","count":0}]:
            print(f"  {s['day']}  {s['label']}  [{s['count']} events]")

        print("\n  ANOMALIES (spikes > mean + 2σ)")
        for a in report["anomalies"] or [{"day":"—","label":"none detected","count":0,"z_score":0}]:
            print(f"  {a['day']}  {a['label']}  z={a.get('z_score',0)}  [{a['count']} events]")

        v = report["velocity"]
        print(f"\n  VELOCITY  this week: {v['this_week_count']} ({v['this_week_daily']}/day)"
              f"  last week: {v['last_week_count']} ({v['last_week_daily']}/day)"
              f"  trend: {v['trend'].upper()}  ({v['delta_pct']:+.1f}%)")

        print(f"\n  Updated: {report['updated_at']}")
        print(f"{'─'*56}\n")
