#!/usr/bin/env python3.12
"""
PALM COMMAND — Tapo SD-card backfill.

Walks every Tapo camera's on-board SD card recordings, downloads any
clip not already in the local event DB, extracts a snapshot, runs
YOLOv8 AI detection + person profiler, and inserts the results.

Usage (inside Docker or on the same LAN as the cameras):
  python3 backfill_tapo.py                       # last 7 days, all cams
  python3 backfill_tapo.py --days 30             # last 30 days
  python3 backfill_tapo.py --start 20260401      # since Apr 1 2026
  python3 backfill_tapo.py --end   20260430      # up through Apr 30
  python3 backfill_tapo.py --cam doorbell        # single camera only
  python3 backfill_tapo.py --dry-run             # show what would import
  python3 backfill_tapo.py --list-dates          # print available dates then exit

Environment / config (same as camera_watcher.py):
  CAMERAS_CONFIG   path to cameras.yaml  (default /config/cameras.yaml)
  MEDIA_DIR        where to store clips  (default /tmp/cams)
  DB_PATH          SQLite path           (default /data/events.db)
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
import yaml
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import ai_engine
import event_db
import profiler

CONFIG_FILE = Path(os.environ.get("CAMERAS_CONFIG", "/config/cameras.yaml"))
MEDIA_DIR   = Path(os.environ.get("MEDIA_DIR",     "/tmp/cams"))

# Tolerance window (seconds) when checking for duplicate events.
# If an event already exists within this range of the recording start_time
# we skip it — avoids re-importing a clip already caught by the live watcher.
DEDUP_WINDOW_S = 30


# ── DB dedup helper ───────────────────────────────────────────────

def _already_imported(cam_id: str, ts: float) -> bool:
    """True if an event for this camera exists within DEDUP_WINDOW_S of ts."""
    import sqlite3
    db_path = Path(os.environ.get("DB_PATH", "/data/events.db"))
    if not db_path.exists():
        return False
    try:
        with sqlite3.connect(str(db_path)) as c:
            row = c.execute(
                "SELECT id FROM events "
                "WHERE camera_id = ? AND ts >= ? AND ts <= ? LIMIT 1",
                (cam_id, ts - DEDUP_WINDOW_S, ts + DEDUP_WINDOW_S),
            ).fetchone()
            return row is not None
    except Exception:
        return False


# ── Per-recording output paths ────────────────────────────────────

def _recording_dir(cam_id: str, start_ts: int) -> Path:
    d = MEDIA_DIR / cam_id / "backfill" / str(start_ts)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── ffmpeg: TS bytes → clip.mp4 + snap.jpg ───────────────────────

def _ffmpeg_ts(ts_data: bytes, out_dir: Path, duration: int) -> tuple[Path | None, Path | None]:
    """
    Convert raw MPEG-TS bytes into a clip and a single JPEG frame.
    Returns (clip_path, snap_path) — either may be None on failure.
    """
    clip = out_dir / "clip.mp4"
    snap = out_dir / "snap.jpg"
    clip_path: Optional[Path] = None
    snap_path: Optional[Path] = None

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "mpegts", "-i", "pipe:0",
             "-c:v", "copy", "-an", "-movflags", "+faststart",
             "-t", str(duration), str(clip)],
            input=ts_data, capture_output=True, timeout=60,
            check=False,
        )
        if clip.exists() and clip.stat().st_size > 10_000:
            clip_path = clip
    except Exception as e:
        print(f"    [ffmpeg clip] {e}", flush=True)

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "mpegts", "-i", "pipe:0",
             "-vframes", "1", "-q:v", "2", "-f", "image2", str(snap)],
            input=ts_data, capture_output=True, timeout=30,
            check=False,
        )
        if snap.exists() and snap.stat().st_size > 0:
            snap_path = snap
    except Exception as e:
        print(f"    [ffmpeg snap] {e}", flush=True)

    return clip_path, snap_path


# ── Download one recording from the camera ────────────────────────

async def _download_recording(
    cam_cfg: dict,
    start_ts: int,
    end_ts: int,
) -> bytes:
    """Stream a recorded clip from the camera SD card. Returns raw MPEG-TS bytes."""
    from pytapo import Tapo
    from pytapo.media_stream._utils import StreamType

    ip       = cam_cfg["ip"]
    pwd      = cam_cfg["password"]
    cloud_pw = cam_cfg.get("cloud_password", pwd)   # use same pw if not specified
    port     = int(cam_cfg.get("port", 8800))

    tapo    = Tapo(ip, "admin", pwd, cloudPassword=cloud_pw)
    session = tapo.getMediaSession(
        stream_type=StreamType.Download,
        start_time=str(start_ts),
    )

    from pytapo.const import EncryptionMethod
    session.port          = port
    session.window_size   = 50

    buf = bytearray()
    try:
        await asyncio.wait_for(session.start(), timeout=10)
        deadline = time.monotonic() + (end_ts - start_ts) + 15
        async for resp in session.transceive(
            # Download request payload (same structure as live preview but type=download)
            '{"type":"request","seq":1,"params":{"playback":{"audio":["default"],'
            '"channels":[0],"resolutions":["HD"],"start_time":"' + str(start_ts) + '",'
            '"end_time":"' + str(end_ts) + '"}},"method":"get"}',
            no_data_timeout=8.0,
        ):
            if resp.mimetype == "video/mp2t" and isinstance(resp.plaintext, bytes):
                buf.extend(resp.plaintext)
            if time.monotonic() >= deadline:
                break
    except Exception as e:
        print(f"    [stream] {e}", flush=True)
    finally:
        try:
            await session.close()
        except Exception:
            pass

    return bytes(buf)


# ── Process one camera ────────────────────────────────────────────

async def backfill_camera(
    cam_cfg: dict,
    start_date: str,
    end_date: str,
    dry_run: bool,
    list_dates: bool,
) -> dict:
    from pytapo import Tapo

    cam_id = cam_cfg["id"]
    ip     = cam_cfg["ip"]
    pwd    = cam_cfg["password"]

    stats = {"cam": cam_id, "found": 0, "imported": 0, "skipped": 0, "errors": 0}

    print(f"\n[{cam_id}] Connecting to {ip}...", flush=True)
    try:
        tapo  = Tapo(ip, "admin", pwd, cloudPassword=pwd)
        dates = tapo.getRecordingsList(start_date=start_date, end_date=end_date)
    except Exception as e:
        print(f"[{cam_id}] Connection failed: {e}", flush=True)
        stats["errors"] += 1
        return stats

    # getRecordingsList returns a list of date strings "YYYYMMDD"
    available: list[str] = []
    if isinstance(dates, list):
        available = [d for d in dates if isinstance(d, str)]
    elif isinstance(dates, dict):
        # Some firmware returns {"playback": {"search_year_utility": [...]}}
        for v in dates.values():
            if isinstance(v, list):
                available.extend(str(x) for x in v)

    available = sorted(available)
    print(f"[{cam_id}] {len(available)} date(s) with recordings: {', '.join(available) or 'none'}", flush=True)

    if list_dates or not available:
        return stats

    for date_str in available:
        print(f"[{cam_id}] Scanning {date_str}...", flush=True)
        try:
            recordings = tapo.getRecordings(date_str)
        except Exception as e:
            print(f"  [{cam_id}] {date_str} getRecordings error: {e}", flush=True)
            stats["errors"] += 1
            continue

        if not isinstance(recordings, list):
            print(f"  [{cam_id}] {date_str} unexpected response: {recordings}", flush=True)
            continue

        for rec in recordings:
            # Each recording has startTime / endTime (unix seconds, camera clock)
            try:
                start_ts = int(rec.get("startTime") or rec.get("start_time", 0))
                end_ts   = int(rec.get("endTime")   or rec.get("end_time",   0))
            except Exception:
                continue

            if not start_ts:
                continue

            duration   = max(end_ts - start_ts, 1) if end_ts > start_ts else 30
            event_ts   = float(start_ts)
            dt_str     = datetime.fromtimestamp(event_ts).strftime("%Y-%m-%d %H:%M:%S")
            stats["found"] += 1

            if _already_imported(cam_id, event_ts):
                print(f"  [{cam_id}] {dt_str} — already in DB, skip", flush=True)
                stats["skipped"] += 1
                continue

            if dry_run:
                print(f"  [{cam_id}] {dt_str} — would import ({duration}s clip)", flush=True)
                stats["imported"] += 1
                continue

            print(f"  [{cam_id}] {dt_str} — downloading {duration}s clip...", flush=True)
            try:
                ts_data = await _download_recording(cam_cfg, start_ts, end_ts)
            except Exception as e:
                print(f"  [{cam_id}] download error: {e}", flush=True)
                stats["errors"] += 1
                continue

            if len(ts_data) < 4096:
                print(f"  [{cam_id}] {dt_str} — too small ({len(ts_data)}b), skip", flush=True)
                stats["errors"] += 1
                continue

            out_dir              = _recording_dir(cam_id, start_ts)
            clip_path, snap_path = _ffmpeg_ts(ts_data, out_dir, duration)

            clip_str = str(clip_path) if clip_path else None
            snap_str = str(snap_path) if snap_path else None

            # AI detection
            detections: list[dict] = []
            if snap_path:
                detections = ai_engine.detect(snap_path)
                if detections:
                    ai_engine.annotate(snap_path, detections)

            # Insert event
            event_id = event_db.insert_event(cam_id, event_ts, clip_str, snap_str)
            if detections:
                event_db.add_detections(event_id, detections)

            # Person profiling
            if snap_path and detections:
                crops = ai_engine.extract_crops(snap_path, detections)
                profiler.match_or_create(cam_id, event_ts, crops, event_id)

            tag_str = ", ".join(
                f"{d['class'].upper()} {int(d['confidence']*100)}%"
                for d in detections
            ) if detections else "no detections"
            print(f"  [{cam_id}] {dt_str} — imported  AI: {tag_str}", flush=True)
            stats["imported"] += 1

    return stats


# ── Config loader ─────────────────────────────────────────────────

def load_cameras() -> list[dict]:
    if not CONFIG_FILE.exists():
        # Fall back to env vars (single camera)
        return [{
            "id":       "doorbell",
            "name":     "Front Door",
            "type":     "tapo",
            "ip":       os.environ.get("TAPO_IP", ""),
            "password": os.environ.get("TAPO_PASSWORD", ""),
            "port":     8800,
        }]
    raw      = CONFIG_FILE.read_text()
    expanded = os.path.expandvars(raw)
    return yaml.safe_load(expanded)["cameras"]


# ── Entry point ───────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill PALM COMMAND DB from Tapo camera SD cards"
    )
    parser.add_argument("--days",       type=int, default=7,
                        help="How many days back to scan (default 7)")
    parser.add_argument("--start",      type=str, default=None,
                        help="Start date YYYYMMDD (overrides --days)")
    parser.add_argument("--end",        type=str, default=None,
                        help="End date YYYYMMDD (default: today)")
    parser.add_argument("--cam",        type=str, default=None,
                        help="Only process this camera id")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Show what would be imported without writing anything")
    parser.add_argument("--list-dates", action="store_true",
                        help="Print available recording dates then exit")
    args = parser.parse_args()

    today      = datetime.now().strftime("%Y%m%d")
    end_date   = args.end or today
    if args.start:
        start_date = args.start
    else:
        start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y%m%d")

    print(f"PALM COMMAND — Tapo SD backfill", flush=True)
    print(f"Date range : {start_date} → {end_date}", flush=True)
    if args.dry_run:
        print("Mode       : DRY RUN (no writes)", flush=True)

    event_db.init_db()

    all_cams = load_cameras()
    tapo_cams = [c for c in all_cams if c.get("type") == "tapo"]
    if args.cam:
        tapo_cams = [c for c in tapo_cams if c["id"] == args.cam]

    if not tapo_cams:
        print("No Tapo cameras found in config (or --cam filter matched nothing).", flush=True)
        sys.exit(1)

    print(f"Cameras    : {', '.join(c['id'] for c in tapo_cams)}", flush=True)

    t0      = time.monotonic()
    results = []
    for cam in tapo_cams:
        r = await backfill_camera(cam, start_date, end_date, args.dry_run, args.list_dates)
        results.append(r)

    # Summary
    elapsed = int(time.monotonic() - t0)
    print("\n" + "─" * 48, flush=True)
    print(f"{'CAMERA':<16}  {'FOUND':>6}  {'IMPORTED':>8}  {'SKIPPED':>7}  {'ERRORS':>6}", flush=True)
    print("─" * 48, flush=True)
    total_f = total_i = total_s = total_e = 0
    for r in results:
        print(f"{r['cam']:<16}  {r['found']:>6}  {r['imported']:>8}  {r['skipped']:>7}  {r['errors']:>6}", flush=True)
        total_f += r["found"]; total_i += r["imported"]
        total_s += r["skipped"]; total_e += r["errors"]
    print("─" * 48, flush=True)
    print(f"{'TOTAL':<16}  {total_f:>6}  {total_i:>8}  {total_s:>7}  {total_e:>6}", flush=True)
    print(f"\nCompleted in {elapsed}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
