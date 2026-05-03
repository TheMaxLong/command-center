#!/usr/bin/env python3.12
"""
PALM COMMAND — Persistent SQLite event store v2.

Schema additions over v1:
  • merge_profiles()        — merge duplicate profiles
  • set_profile_label()     — persist custom name for a profile
  • get_events_in_range()   — time-range query for velocity tracking
  • get_detection_summary() now supports weeks=0 (last 24h)
"""
import json, os, sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

def _default_db_path() -> Path:
    """Return a writable DB path, falling back to /tmp if /data doesn't exist."""
    env = os.environ.get("DB_PATH", "")
    if env:
        return Path(env)
    data_dir = Path("/data")
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        test = data_dir / ".write_test"
        test.touch(); test.unlink()
        return data_dir / "events.db"
    except Exception:
        return Path("/tmp/palm_command_events.db")

DB_PATH = _default_db_path()


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA foreign_keys=ON")
    return c


def init_db() -> None:
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id   TEXT    NOT NULL,
                ts          REAL    NOT NULL,
                day_of_week INTEGER NOT NULL,
                hour        INTEGER NOT NULL,
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
                bbox        TEXT,
                instance    INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_det_event ON detections(event_id);
            CREATE INDEX IF NOT EXISTS idx_det_class ON detections(class_name);

            CREATE TABLE IF NOT EXISTS profiles (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                first_seen  REAL    NOT NULL,
                last_seen   REAL    NOT NULL,
                sightings   INTEGER NOT NULL DEFAULT 1,
                cameras     TEXT    NOT NULL DEFAULT '[]',
                embedding   TEXT    NOT NULL,
                thumb       BLOB,
                label       TEXT
            );

            CREATE TABLE IF NOT EXISTS profile_sightings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id  INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
                ts          REAL    NOT NULL,
                cam_id      TEXT    NOT NULL,
                event_id    INTEGER REFERENCES events(id)
            );
            CREATE INDEX IF NOT EXISTS idx_ps_profile ON profile_sightings(profile_id, ts);

            -- intel_alerts: pinned alerts surfaced to dashboard
            CREATE TABLE IF NOT EXISTS intel_alerts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                type        TEXT    NOT NULL,
                severity    TEXT    NOT NULL DEFAULT 'info',
                message     TEXT    NOT NULL,
                camera_id   TEXT,
                ts          REAL    NOT NULL,
                read        INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_alerts_ts ON intel_alerts(ts);

            -- manual_scans: field app plate/face scan log
            CREATE TABLE IF NOT EXISTS manual_scans (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_type     TEXT    NOT NULL,
                plate         TEXT,
                confidence    REAL,
                watchlist_hit INTEGER DEFAULT 0,
                fbi_match     INTEGER,
                match_name    TEXT,
                timestamp     TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ms_ts ON manual_scans(timestamp);

            -- Migration: add instance column to detections if missing
            -- (safe to run on existing databases)
        """)
        # Safe column-add migrations
        try:
            c.execute("ALTER TABLE detections ADD COLUMN instance INTEGER NOT NULL DEFAULT 0")
        except Exception:
            pass


# ── Events ────────────────────────────────────────────────────────

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
        (
            event_id,
            d["class"],
            d["confidence"],
            json.dumps(d.get("bbox")),
            d.get("instance", 0),
        )
        for d in detections
    ]
    with _conn() as c:
        c.executemany(
            "INSERT INTO detections (event_id, class_name, confidence, bbox, instance) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )


# ── Events: read ─────────────────────────────────────────────────

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


def get_events_in_range(
    ts_from: float,
    ts_to: float,
    camera_id: Optional[str] = None,
) -> list[dict]:
    q = "SELECT id, camera_id, ts FROM events WHERE ts >= ? AND ts <= ?"
    params: list = [ts_from, ts_to]
    if camera_id:
        q += " AND camera_id = ?"
        params.append(camera_id)
    with _conn() as c:
        return [dict(r) for r in c.execute(q, params)]


def get_hourly_heatmap(
    camera_id: Optional[str] = None,
    weeks: int = 5,
) -> list[dict]:
    import time
    if weeks == 0:
        cutoff = time.time() - 86400   # last 24h
    else:
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
    import time
    if weeks == 0:
        cutoff = time.time() - 86400
    else:
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


# ── Profiles: write ───────────────────────────────────────────────

def create_profile(
    cam_id: str,
    ts: float,
    embedding: list[float],
    thumb_bytes: Optional[bytes] = None,
) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO profiles (first_seen, last_seen, sightings, cameras, embedding, thumb) "
            "VALUES (?, ?, 1, ?, ?, ?)",
            (ts, ts, json.dumps([cam_id]), json.dumps(embedding), thumb_bytes),
        )
        pid = cur.lastrowid
        c.execute(
            "INSERT INTO profile_sightings (profile_id, ts, cam_id) VALUES (?, ?, ?)",
            (pid, ts, cam_id),
        )
        return pid


def update_profile_sighting(
    profile_id: int,
    ts: float,
    cam_id: str,
    new_embedding: list[float],
    event_id: Optional[int] = None,
) -> None:
    with _conn() as c:
        row = c.execute(
            "SELECT cameras FROM profiles WHERE id = ?", (profile_id,)
        ).fetchone()
        cameras_list: list[str] = json.loads(row["cameras"]) if row else []
        if cam_id not in cameras_list:
            cameras_list.append(cam_id)
        c.execute(
            "UPDATE profiles SET last_seen = ?, sightings = sightings + 1, "
            "cameras = ?, embedding = ? WHERE id = ?",
            (ts, json.dumps(cameras_list), json.dumps(new_embedding), profile_id),
        )
        c.execute(
            "INSERT INTO profile_sightings (profile_id, ts, cam_id, event_id) "
            "VALUES (?, ?, ?, ?)",
            (profile_id, ts, cam_id, event_id),
        )


def set_profile_label(profile_id: int, label: str) -> bool:
    """Persist a human-readable custom name for a profile. Returns success."""
    with _conn() as c:
        c.execute("UPDATE profiles SET label = ? WHERE id = ?", (label, profile_id))
        return c.execute("SELECT changes()").fetchone()[0] > 0


def merge_profiles(
    keep_id: int,
    drop_id: int,
    merged_cameras: list[str],
    merged_embedding: list[float],
) -> None:
    """
    Merge drop_id into keep_id:
      - Reassign all profile_sightings rows from drop_id to keep_id
      - Add drop's sightings count to keep
      - Update cameras + embedding on keep
      - Delete drop profile
    """
    with _conn() as c:
        # Reassign sightings
        c.execute(
            "UPDATE profile_sightings SET profile_id = ? WHERE profile_id = ?",
            (keep_id, drop_id),
        )
        # Get sightings counts
        drop_row = c.execute(
            "SELECT sightings FROM profiles WHERE id = ?", (drop_id,)
        ).fetchone()
        drop_count = drop_row["sightings"] if drop_row else 0
        # Merge onto keep
        c.execute(
            "UPDATE profiles SET "
            "sightings = sightings + ?, cameras = ?, embedding = ? "
            "WHERE id = ?",
            (drop_count, json.dumps(merged_cameras), json.dumps(merged_embedding), keep_id),
        )
        # Remove the merged profile
        c.execute("DELETE FROM profiles WHERE id = ?", (drop_id,))


# ── Profiles: read ────────────────────────────────────────────────

def get_all_profiles() -> list[dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT id, first_seen, last_seen, sightings, cameras, embedding, thumb, label "
            "FROM profiles ORDER BY last_seen DESC"
        )]


def get_profile(profile_id: int) -> Optional[dict]:
    with _conn() as c:
        row = c.execute(
            "SELECT id, first_seen, last_seen, sightings, cameras, embedding, thumb, label "
            "FROM profiles WHERE id = ?", (profile_id,)
        ).fetchone()
        return dict(row) if row else None


def get_profile_thumb(profile_id: int) -> Optional[bytes]:
    with _conn() as c:
        row = c.execute(
            "SELECT thumb FROM profiles WHERE id = ?", (profile_id,)
        ).fetchone()
        return row["thumb"] if row else None


def get_profile_sightings(profile_id: int) -> list[dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT ts, cam_id, event_id FROM profile_sightings "
            "WHERE profile_id = ? ORDER BY ts DESC",
            (profile_id,),
        )]


# ── Intel alerts ──────────────────────────────────────────────────

def insert_alert(
    type_: str,
    severity: str,
    message: str,
    camera_id: Optional[str],
    ts: float,
) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO intel_alerts (type, severity, message, camera_id, ts) "
            "VALUES (?, ?, ?, ?, ?)",
            (type_, severity, message, camera_id, ts),
        )
        return cur.lastrowid


def get_alerts(limit: int = 20, unread_only: bool = False) -> list[dict]:
    q = "SELECT * FROM intel_alerts"
    if unread_only:
        q += " WHERE read = 0"
    q += " ORDER BY ts DESC LIMIT ?"
    with _conn() as c:
        return [dict(r) for r in c.execute(q, (limit,))]


def mark_alert_read(alert_id: int) -> None:
    with _conn() as c:
        c.execute("UPDATE intel_alerts SET read = 1 WHERE id = ?", (alert_id,))


# ── Field Scan (phone app) ────────────────────────────────────────

def log_manual_scan(
    scan_type: str,
    plate: Optional[str] = None,
    confidence: Optional[float] = None,
    watchlist_hit: bool = False,
    fbi_match: Optional[bool] = None,
    match_name: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> None:
    ts = timestamp or datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _conn() as c:
        c.execute(
            "INSERT INTO manual_scans "
            "(scan_type, plate, confidence, watchlist_hit, fbi_match, match_name, timestamp) "
            "VALUES (?,?,?,?,?,?,?)",
            (scan_type, plate, confidence, int(watchlist_hit),
             int(fbi_match) if fbi_match is not None else None,
             match_name, ts),
        )


def get_manual_scans(limit: int = 50) -> dict:
    with _conn() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT scan_type, plate, confidence, watchlist_hit, "
            "fbi_match, match_name, timestamp "
            "FROM manual_scans ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    scans = [dict(r) for r in rows]
    for s in scans:
        s["watchlist_hit"] = bool(s["watchlist_hit"])
        if s["fbi_match"] is not None:
            s["fbi_match"] = bool(s["fbi_match"])
    return {"scans": scans, "count": len(scans)}
