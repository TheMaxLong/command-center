#!/usr/bin/env python3.12
"""
PALM COMMAND — 5-week trend analyzer.

Queries the event DB to surface:
  - Hourly activity heatmap (day × hour)
  - Inferred schedule: peak hours per day that exceed mean + 1σ
  - Top detected classes with frequencies
  - Anomaly spikes: (day, hour) cells beyond mean + 2σ globally

Used by camera_watcher.py's /trends endpoint, and can be run
standalone to print a summary to stdout.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import event_db

DAYS = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]


def analyze(camera_id: Optional[str] = None, weeks: int = 5) -> dict:
    """
    Return a full trend report dict ready for JSON serialisation.

    Keys:
      weeks, camera_id, heatmap, schedule, detections, anomalies, updated_at
    """
    heatmap    = event_db.get_hourly_heatmap(camera_id, weeks)
    detections = event_db.get_detection_summary(camera_id, weeks)
    schedule   = _infer_schedule(heatmap)
    anomalies  = _detect_anomalies(heatmap)

    return {
        "weeks":      weeks,
        "camera_id":  camera_id,
        "heatmap":    heatmap,
        "schedule":   schedule,
        "detections": detections,
        "anomalies":  anomalies,
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    }


# ── Internal helpers ──────────────────────────────────────────────

def _infer_schedule(heatmap: list[dict]) -> list[dict]:
    """
    Find hours per day whose event count exceeds mean + 1σ for that day.
    These represent recurring patterns (likely scheduled activity).
    """
    if not heatmap:
        return []

    # Build full 7×24 grid (missing cells = 0)
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
                    "day":   DAYS[day],
                    "hour":  hour,
                    "label": f"{hour:02d}:00",
                    "count": c,
                })

    return sorted(schedule, key=lambda x: (DAYS.index(x["day"]), x["hour"]))


def _detect_anomalies(heatmap: list[dict]) -> list[dict]:
    """
    Flag (day, hour) cells whose count is > mean + 2σ globally.
    These are unusual spikes worth investigating.
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
                "hour":    row["hour"],
                "label":   f"{row['hour']:02d}:00",
                "count":   row["count"],
                "z_score": round(z, 2),
            })

    return sorted(anomalies, key=lambda x: -x["z_score"])


# ── CLI summary ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    ap = argparse.ArgumentParser(description="Print 5-week trend report")
    ap.add_argument("--camera", default=None, help="Camera ID filter")
    ap.add_argument("--weeks",  type=int, default=5)
    ap.add_argument("--json",   action="store_true", help="Output raw JSON")
    args = ap.parse_args()

    event_db.init_db()
    report = analyze(camera_id=args.camera, weeks=args.weeks)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"\n{'─'*54}")
        print(f"  PALM COMMAND · Trend Report · Last {args.weeks} weeks")
        if args.camera:
            print(f"  Camera: {args.camera}")
        print(f"{'─'*54}")

        print("\n  TOP DETECTIONS")
        for d in report["detections"][:8]:
            bar = "█" * min(20, d["count"])
            print(f"  {d['class_name']:<12} {d['count']:>4}x  {bar}  ({d['avg_conf']*100:.0f}% avg conf)")

        print("\n  RECURRING SCHEDULE (peaks > mean + 1σ per day)")
        if report["schedule"]:
            for s in report["schedule"]:
                print(f"  {s['day']}  {s['label']}  [{s['count']} events]")
        else:
            print("  — not enough data yet —")

        print("\n  ANOMALIES (spikes > mean + 2σ)")
        if report["anomalies"]:
            for a in report["anomalies"]:
                print(f"  {a['day']}  {a['label']}  z={a['z_score']}  [{a['count']} events]")
        else:
            print("  — none detected —")

        print(f"\n  Updated: {report['updated_at']}")
        print(f"{'─'*54}\n")
