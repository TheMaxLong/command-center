"""
COMMAND CENTER — Anomaly baseline drift detection.

For each (day_of_week, hour) cell, compute the historical mean + stddev of
motion event counts over the past N weeks. Score each current-week cell
as a z-score against that baseline. Surfaces statistical outliers in the
operator's own rhythm — "this Tuesday at 3:15am is 4.2σ outside normal."

Designed to land cleanly on the existing event_db schema. No new tables.

Endpoint wiring lives in camera_watcher.py.
"""
from __future__ import annotations

import math
import time
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


DB_PATH = "/data/events.db"


def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=4.0)
    conn.row_factory = sqlite3.Row
    return conn


def _cell_counts(conn: sqlite3.Connection, ts_min: float, ts_max: float,
                 camera_id: Optional[str] = None) -> dict[tuple[int, int, str], int]:
    """Return {(dow, hour, week_key): event_count} between ts_min and ts_max.

    week_key is ISO year-week (e.g. "2026-W21") so we can split baseline weeks
    from the current week.
    """
    sql = ["SELECT ts, day_of_week, hour, camera_id FROM events WHERE ts BETWEEN ? AND ?"]
    args: list[Any] = [ts_min, ts_max]
    if camera_id:
        sql.append("AND camera_id = ?")
        args.append(camera_id)
    out: dict[tuple[int, int, str], int] = {}
    for row in conn.execute(" ".join(sql), args):
        dow = int(row["day_of_week"] or 0)  # 0..6 (Mon..Sun in our schema)
        hr  = int(row["hour"] or 0)
        wk = datetime.fromtimestamp(row["ts"], tz=timezone.utc).strftime("%G-W%V")
        out[(dow, hr, wk)] = out.get((dow, hr, wk), 0) + 1
    return out


def baseline_grid(weeks: int = 4, camera_id: Optional[str] = None) -> dict:
    """Compute the 7x24 anomaly heatmap.

    Returns: {
        weeks_window:    int,
        baseline_n_weeks: int,        # weeks contributing to baseline
        current_week:    str,         # ISO week key
        sample_count:    int,         # total events used
        grid: [
            { dow, hour, mean, std, count_current,
              z_score, severity }  × 168 cells
        ],
        top_anomalies: [...]          # cells with |z|>=2 sorted by z desc
    }
    """
    now    = time.time()
    cutoff = now - weeks * 7 * 24 * 3600

    try:
        conn = _open_db()
    except sqlite3.Error as e:
        return {"error": f"db unavailable: {e}"}

    try:
        cells = _cell_counts(conn, cutoff, now, camera_id)
    finally:
        conn.close()

    if not cells:
        return {
            "weeks_window": weeks,
            "baseline_n_weeks": 0,
            "current_week": datetime.now(tz=timezone.utc).strftime("%G-W%V"),
            "sample_count": 0,
            "grid": [],
            "top_anomalies": [],
            "note": "no events in window — baseline needs ≥1 full week",
        }

    current_week = datetime.now(tz=timezone.utc).strftime("%G-W%V")

    # Per-cell aggregations: for each (dow, hour), collect counts per week.
    # Baseline = weeks other than current. Current = the current week.
    cell_history: dict[tuple[int, int], list[int]] = {}
    cell_current: dict[tuple[int, int], int] = {}
    weeks_seen: set[str] = set()
    sample_count = 0
    for (dow, hr, wk), count in cells.items():
        sample_count += count
        weeks_seen.add(wk)
        if wk == current_week:
            cell_current[(dow, hr)] = cell_current.get((dow, hr), 0) + count
        else:
            cell_history.setdefault((dow, hr), []).append(count)

    grid = []
    for dow in range(7):
        for hr in range(24):
            hist = cell_history.get((dow, hr), [])
            curr = cell_current.get((dow, hr), 0)
            # Pad history with zeros for weeks where this cell had no events.
            # Baseline weeks count = total weeks seen − 1 (the current one)
            baseline_weeks_total = max(0, len(weeks_seen) - (1 if current_week in weeks_seen else 0))
            zero_padding = baseline_weeks_total - len(hist)
            if zero_padding > 0:
                hist = hist + [0] * zero_padding
            n = len(hist)
            if n == 0:
                grid.append({"dow": dow, "hour": hr, "mean": 0, "std": 0,
                             "count_current": curr, "z_score": None,
                             "severity": "no_baseline"})
                continue
            mean = sum(hist) / n
            if n > 1:
                var = sum((x - mean) ** 2 for x in hist) / (n - 1)
                std = math.sqrt(var)
            else:
                std = 0.0
            # z = (current - mean) / std ; if std==0 use a tiny epsilon
            if std < 1e-6:
                z = (curr - mean) * 10  # magnify when baseline is flat — any
                                        # nonzero current is anomalous
            else:
                z = (curr - mean) / std
            if abs(z) >= 3:
                sev = "outlier"
            elif abs(z) >= 2:
                sev = "anomaly"
            elif abs(z) >= 1:
                sev = "elevated"
            else:
                sev = "normal"
            grid.append({
                "dow": dow, "hour": hr,
                "mean": round(mean, 2),
                "std": round(std, 2),
                "count_current": curr,
                "z_score": round(z, 2),
                "severity": sev,
            })

    top = sorted(
        (c for c in grid if c.get("z_score") is not None and abs(c["z_score"]) >= 1.5),
        key=lambda c: -abs(c["z_score"]),
    )[:8]

    baseline_n_weeks = max(0, len(weeks_seen) - (1 if current_week in weeks_seen else 0))

    return {
        "weeks_window": weeks,
        "baseline_n_weeks": baseline_n_weeks,
        "current_week": current_week,
        "sample_count": sample_count,
        "grid": grid,
        "top_anomalies": top,
    }


def cell_events(dow: int, hour: int, days: int = 28,
                camera_id: Optional[str] = None, limit: int = 30) -> list[dict]:
    """Return events that match the (dow, hour) cell over the last N days."""
    cutoff = time.time() - days * 24 * 3600
    sql = "SELECT id, ts, camera_id, day_of_week, hour, snap_path FROM events " \
          "WHERE ts >= ? AND day_of_week = ? AND hour = ?"
    args: list[Any] = [cutoff, dow, hour]
    if camera_id:
        sql += " AND camera_id = ?"
        args.append(camera_id)
    sql += " ORDER BY ts DESC LIMIT ?"
    args.append(limit)

    try:
        conn = _open_db()
        rows = [dict(r) for r in conn.execute(sql, args)]
        conn.close()
        return rows
    except sqlite3.Error:
        return []
