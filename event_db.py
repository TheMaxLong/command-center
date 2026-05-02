#!/usr/bin/env python3.12
"""
Persistent SQLite event store for PALM COMMAND.

Schema:
  events     – one row per motion event (any camera)
  detections – one row per detected object within an event
"""
import json, os, sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(os.environ.get("DB_PATH", "/data/events.db"))


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id   TEXT    NOT NULL,
                ts          REAL    NOT NULL,
                day_of_week INTEGER NOT NULL,   -- 0=Mon … 6=Sun (Python weekday)
                hour        INTEGER NOT NULL,   -- 0-23 UTC
                clip_path   TEXT,
                snap_path   TEXT,
                created_at  TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_camera_ts ON events(camera_id, ts);
            CREATE INDEX IF NOT EXISTS idx_events_ts         ON events(ts);

            CREATE TABLE IF NOT EXISTS detections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id    INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
                class_name  TEXT    NOT NULL,
                confidence  REAL    NOT NULL,
                bbox        TEXT            -- JSON [x1,y1,x2,y2]
            );
            CREATE INDEX IF NOT EXISTS idx_det_event ON detections(event_id);
            CREATE INDEX IF NOT EXISTS idx_det_class ON detections(class_name);
        """)


# ── Write ──────────────────────────────────────────────────────────

def insert_event(
    camera_id: str,
    ts: float,
    clip_path: Optional[str],
    snap_path: Optional[str],
) -> int:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO events "
            "(camera_id, ts, day_of_week, hour, clip_path, snap_path, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (camera_id, ts, dt.weekday(), dt.hour, clip_path, snap_path, dt.isoformat()),
        )
        return cur.lastrowid


def add_detections(event_id: int, detections: list[dict]) -> None:
    rows = [
        (event_id, d["class"], d["confidence"], json.dumps(d.get("bbox")))
        for d in detections
    ]
    with _conn() as c:
        c.executemany(
            "INSERT INTO detections (event_id, class_name, confidence, bbox) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )


# ── Read ───────────────────────────────────────────────────────────

def get_recent_events(
    camera_id: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    q = (
        "SELECT e.id, e.camera_id, e.ts, e.day_of_week, e.hour, "
        "e.clip_path, e.snap_path, e.created_at, "
        "GROUP_CONCAT(d.class_name || ':' || ROUND(d.confidence, 2)) AS tags "
        "FROM events e "
        "LEFT JOIN detections d ON d.event_id = e.id "
    )
    params: list = []
    if camera_id:
        q += "WHERE e.camera_id = ? "
        params.append(camera_id)
    q += "GROUP BY e.id ORDER BY e.ts DESC LIMIT ?"
    params.append(limit)
    with _conn() as c:
        return [dict(r) for r in c.execute(q, params)]


def get_hourly_heatmap(
    camera_id: Optional[str] = None,
    weeks: int = 5,
) -> list[dict]:
    """Event counts by (day_of_week × hour) over the last N weeks."""
    import time
    cutoff = time.time() - weeks * 7 * 86400
    q = "SELECT day_of_week, hour, COUNT(*) AS count FROM events WHERE ts >= ?"
    params: list = [cutoff]
    if camera_id:
        q += " AND camera_id = ?"
        params.append(camera_id)
    q += " GROUP BY day_of_week, hour ORDER BY day_of_week, hour"
    with _conn() as c:
        return [dict(r) for r in c.execute(q, params)]


def get_detection_summary(
    camera_id: Optional[str] = None,
    weeks: int = 5,
) -> list[dict]:
    """Top detected classes with counts and average confidence."""
    import time
    cutoff = time.time() - weeks * 7 * 86400
    q = (
        "SELECT d.class_name, COUNT(*) AS count, "
        "ROUND(AVG(d.confidence), 3) AS avg_conf "
        "FROM detections d "
        "JOIN events e ON e.id = d.event_id "
        "WHERE e.ts >= ? "
    )
    params: list = [cutoff]
    if camera_id:
        q += "AND e.camera_id = ? "
        params.append(camera_id)
    q += "GROUP BY d.class_name ORDER BY count DESC"
    with _conn() as c:
        return [dict(r) for r in c.execute(q, params)]
