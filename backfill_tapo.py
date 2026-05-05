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
import json
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

# If an event already exists within this window of a recording's start_time, skip it.
DEDUP_WINDOW_S = 30

# ── Active backfill jobs (cam_id → status dict) — used by API endpoint ──
_jobs: dict[str, dict] = {}


# ── DB dedup ──────────────────────────────────────────────────────────────

def _already_imported(cam_id: str, ts: float) -> bool:
    import sqlite3
    db_path = Path(os.environ.get("DB_PATH", "/data/events.db"))
    if not db_path.exists():
        return False
    try:
        with sqlite3.connect(str(db_path)) as c:
            row = c.execute(
                "SELECT id FROM events "
                "WHERE camera_id=? AND ts>=? AND ts<=? LIMIT 1",
                (cam_id, ts - DEDUP_WINDOW_S, ts + DEDUP_WINDOW_S),
            ).fetchone()
            return row is not None
    except Exception:
        return False


# ── Output paths ───────────────────────────────────────────────────────────

def _out_dir(cam_id: str, start_ts: int) -> Path:
    d = MEDIA_DIR / cam_id / "backfill" / str(start_ts)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── ffmpeg ─────────────────────────────────────────────────────────────────

def _ffmpeg_ts(
    ts_data: bytes, out_dir: Path, duration: int
) -> tuple[Optional[Path], Optional[Path]]:
    clip = out_dir / "clip.mp4"
    snap = out_dir / "snap.jpg"
    clip_ok = snap_ok = False
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "mpegts", "-i", "pipe:0",
             "-c:v", "copy", "-an", "-movflags", "+faststart",
             "-t", str(duration), str(clip)],
            input=ts_data, capture_output=True, timeout=60, check=False,
        )
        clip_ok = clip.exists() and clip.stat().st_size > 10_000
    except Exception as e:
        print(f"    [ffmpeg clip] {e}", flush=True)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "mpegts", "-i", "pipe:0",
             "-vframes", "1", "-q:v", "2", "-f", "image2", str(snap)],
            input=ts_data, capture_output=True, timeout=30, check=False,
        )
        snap_ok = snap.exists() and snap.stat().st_size > 0
    except Exception as e:
        print(f"    [ffmpeg snap] {e}", flush=True)
    return (clip if clip_ok else None), (snap if snap_ok else None)


# ── pytapo: list recordings ────────────────────────────────────────────────

def _list_dates(ip: str, pwd: str, start_date: str, end_date: str) -> list[str]:
    """Return sorted list of YYYYMMDD date strings that have recordings."""
    try:
        from pytapo import Tapo
        tapo   = Tapo(ip, "admin", pwd, cloudPassword=pwd)
        result = tapo.getRecordingsList(start_date)
    except Exception as e:
        print(f"  [backfill] getRecordingsList failed: {e}", flush=True)
        return []

    dates: list[str] = []
    if isinstance(result, list):
        dates = [str(d) for d in result if d]
    elif isinstance(result, dict):
        # firmware variants: {"playback": {"search_year_utility": [...]}}
        # or {"recordings_list": [{"date": "YYYYMMDD"}, ...]}
        for val in result.values():
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, str) and len(item) == 8 and item.isdigit():
                        dates.append(item)
                    elif isinstance(item, dict) and "date" in item:
                        dates.append(str(item["date"]))
            elif isinstance(val, dict):
                for inner in val.values():
                    if isinstance(inner, list):
                        for item in inner:
                            if isinstance(item, str) and len(item) == 8 and item.isdigit():
                                dates.append(item)

    # Filter to requested date range
    dates = [d for d in dates if start_date <= d <= end_date]
    return sorted(set(dates))


def _list_recordings(ip: str, pwd: str, date_str: str) -> list[dict]:
    """Return list of recording segments for a date. Each has startTime/endTime."""
    try:
        from pytapo import Tapo
        tapo   = Tapo(ip, "admin", pwd, cloudPassword=pwd)
        result = tapo.getRecordings(date_str)
    except Exception as e:
        print(f"  [backfill] getRecordings({date_str}) failed: {e}", flush=True)
        return []

    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for val in result.values():
            if isinstance(val, list):
                return val
    return []


# ── pytapo: download one clip via HttpMediaSession ─────────────────────────

async def _download_clip(cam_cfg: dict, start_ts: int, end_ts: int) -> bytes:
    """Stream a recorded clip from the SD card. Returns raw MPEG-TS bytes."""
    from pytapo import HttpMediaSession
    from pytapo.const import EncryptionMethod

    ip   = cam_cfg["ip"]
    port = int(cam_cfg.get("port", 8800))
    pwd  = cam_cfg["password"]
    dur  = max(end_ts - start_ts, 5)

    # Playback request — same wire format as PREVIEW_REQ but uses "playback" key
    req = json.dumps({
        "type": "request", "seq": 1,
        "params": {
            "playback": {
                "audio":      ["default"],
                "channels":   [0],
                "resolutions": ["HD"],
                "start_time": str(start_ts),
                "end_time":   str(end_ts),
            },
            "method": "get",
        },
    })

    session = HttpMediaSession(
        ip=ip, cloud_password=pwd, super_secret_key="",
        encryptionMethod=EncryptionMethod.SHA256, port=port, window_size=50,
    )
    buf = bytearray()
    try:
        await asyncio.wait_for(session.start(), timeout=12)
        deadline = time.monotonic() + dur + 10
        async for resp in session.transceive(req, no_data_timeout=6.0):
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


# ── Process one camera ─────────────────────────────────────────────────────

async def backfill_camera(
    cam_cfg: dict,
    start_date: str,
    end_date: str,
    dry_run: bool,
    list_dates_only: bool,
    job_id: Optional[str] = None,
) -> dict:
    cam_id = cam_cfg["id"]
    ip     = cam_cfg["ip"]
    pwd    = cam_cfg["password"]

    stats = {"cam": cam_id, "found": 0, "imported": 0, "skipped": 0, "errors": 0, "done": False}
    if job_id:
        _jobs[job_id] = {**stats, "status": "connecting", "log": []}

    def _log(msg: str) -> None:
        print(msg, flush=True)
        if job_id and job_id in _jobs:
            _jobs[job_id]["log"].append(msg)

    _log(f"\n[{cam_id}] Connecting to {ip}...")
    if job_id:
        _jobs[job_id]["status"] = "listing"

    dates = _list_dates(ip, pwd, start_date, end_date)
    _log(f"[{cam_id}] {len(dates)} date(s) with recordings: {', '.join(dates) or 'none'}")

    if list_dates_only or not dates:
        stats["done"] = True
        if job_id:
            _jobs[job_id] = {**_jobs[job_id], **stats, "status": "done"}
        return stats

    for date_str in dates:
        _log(f"[{cam_id}] Scanning {date_str}...")
        segments = _list_recordings(ip, pwd, date_str)
        if not segments:
            _log(f"  [{cam_id}] {date_str} — no segments returned")
            continue

        _log(f"  [{cam_id}] {date_str} — {len(segments)} segment(s)")
        if job_id:
            _jobs[job_id]["status"] = f"downloading {date_str}"

        for seg in segments:
            try:
                start_ts = int(
                    seg.get("startTime") or seg.get("start_time") or
                    seg.get("StartTime") or 0
                )
                end_ts = int(
                    seg.get("endTime") or seg.get("end_time") or
                    seg.get("EndTime") or 0
                )
            except Exception:
                stats["errors"] += 1
                continue

            if not start_ts:
                continue

            duration = max(end_ts - start_ts, 5) if end_ts > start_ts else 15
            event_ts = float(start_ts)
            dt_str   = datetime.fromtimestamp(event_ts).strftime("%Y-%m-%d %H:%M:%S")
            stats["found"] += 1
            if job_id:
                _jobs[job_id]["found"] = stats["found"]

            if _already_imported(cam_id, event_ts):
                _log(f"  [{cam_id}] {dt_str} — already in DB, skip")
                stats["skipped"] += 1
                if job_id:
                    _jobs[job_id]["skipped"] = stats["skipped"]
                continue

            if dry_run:
                _log(f"  [{cam_id}] {dt_str} — would import ({duration}s)")
                stats["imported"] += 1
                if job_id:
                    _jobs[job_id]["imported"] = stats["imported"]
                continue

            _log(f"  [{cam_id}] {dt_str} — downloading {duration}s...")
            try:
                data = await _download_clip(cam_cfg, start_ts, end_ts)
            except Exception as e:
                _log(f"  [{cam_id}] download error: {e}")
                stats["errors"] += 1
                if job_id:
                    _jobs[job_id]["errors"] = stats["errors"]
                continue

            if len(data) < 4096:
                _log(f"  [{cam_id}] {dt_str} — too small ({len(data)}b), skip")
                stats["errors"] += 1
                if job_id:
                    _jobs[job_id]["errors"] = stats["errors"]
                continue

            out          = _out_dir(cam_id, start_ts)
            clip_p, snap_p = _ffmpeg_ts(data, out, duration)
            clip_str     = str(clip_p) if clip_p else None
            snap_str     = str(snap_p) if snap_p else None

            detections: list[dict] = []
            if snap_p:
                detections = ai_engine.detect(snap_p)
                if detections:
                    ai_engine.annotate(snap_p, detections)

            event_id = event_db.insert_event(cam_id, event_ts, clip_str, snap_str)
            if detections:
                event_db.add_detections(event_id, detections)
            # Backfill clips live in backfill/ subdir — register as archived so
            # they appear in the STORAGE tab and can be star-protected from purge
            event_db.set_event_archived_paths(event_id, clip_str, snap_str)

            if snap_p and detections:
                crops = ai_engine.extract_crops(snap_p, detections)
                profiler.match_or_create(cam_id, event_ts, crops, event_id)

            tag_str = ", ".join(
                f"{d['class'].upper()} {int(d['confidence']*100)}%"
                for d in detections
            ) if detections else "no detections"
            _log(f"  [{cam_id}] {dt_str} — imported  AI: {tag_str}")
            stats["imported"] += 1
            if job_id:
                _jobs[job_id]["imported"] = stats["imported"]

            await asyncio.sleep(0.5)  # breathe between downloads

    stats["done"] = True
    if job_id:
        _jobs[job_id] = {**_jobs[job_id], **stats, "status": "done"}
    return stats


# ── Called from camera_watcher API endpoint ────────────────────────────────

def start_backfill_job(cam_cfg: dict, days: int = 7) -> str:
    """Launch a backfill in a background asyncio task. Returns job_id."""
    import threading, uuid

    today      = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    job_id     = f"{cam_cfg['id']}_{int(time.time())}"

    _jobs[job_id] = {
        "cam": cam_cfg["id"], "status": "queued",
        "found": 0, "imported": 0, "skipped": 0, "errors": 0,
        "done": False, "log": [],
    }

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            event_db.init_db()
            loop.run_until_complete(
                backfill_camera(cam_cfg, start_date, today,
                                dry_run=False, list_dates_only=False,
                                job_id=job_id)
            )
        finally:
            loop.close()

    threading.Thread(target=_run, daemon=True).start()
    return job_id


def get_job_status(job_id: str) -> Optional[dict]:
    return _jobs.get(job_id)


def list_jobs() -> list[dict]:
    return [{"job_id": k, **v} for k, v in _jobs.items()]


# ── CLI entry point ────────────────────────────────────────────────────────

def _load_cameras() -> list[dict]:
    if not CONFIG_FILE.exists():
        return [{
            "id": "doorbell", "name": "Front Door", "type": "tapo",
            "ip": os.environ.get("TAPO_IP", ""),
            "password": os.environ.get("TAPO_PASSWORD", ""),
            "port": 8800,
        }]
    raw = CONFIG_FILE.read_text()
    return yaml.safe_load(os.path.expandvars(raw))["cameras"]


async def main() -> None:
    parser = argparse.ArgumentParser(description="PALM COMMAND — Tapo SD backfill")
    parser.add_argument("--days",       type=int, default=7)
    parser.add_argument("--start",      type=str, default=None)
    parser.add_argument("--end",        type=str, default=None)
    parser.add_argument("--cam",        type=str, default=None)
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--list-dates", action="store_true")
    args = parser.parse_args()

    today      = datetime.now().strftime("%Y%m%d")
    end_date   = args.end or today
    start_date = args.start or (datetime.now() - timedelta(days=args.days)).strftime("%Y%m%d")

    print(f"PALM COMMAND — Tapo SD backfill", flush=True)
    print(f"Range: {start_date} → {end_date}{'  [DRY RUN]' if args.dry_run else ''}", flush=True)

    event_db.init_db()

    cams = [c for c in _load_cameras() if c.get("type") == "tapo"]
    if args.cam:
        cams = [c for c in cams if c["id"] == args.cam]
    if not cams:
        print("No Tapo cameras found.", flush=True)
        sys.exit(1)

    print(f"Cameras: {', '.join(c['id'] for c in cams)}", flush=True)

    t0      = time.monotonic()
    results = []
    for cam in cams:
        r = await backfill_camera(cam, start_date, end_date,
                                  args.dry_run, args.list_dates)
        results.append(r)

    elapsed = int(time.monotonic() - t0)
    print("\n" + "─" * 52, flush=True)
    print(f"{'CAMERA':<16}  {'FOUND':>5}  {'IMPORTED':>8}  {'SKIPPED':>7}  {'ERR':>5}", flush=True)
    print("─" * 52, flush=True)
    tf = ti = ts_ = te = 0
    for r in results:
        print(f"{r['cam']:<16}  {r['found']:>5}  {r['imported']:>8}  {r['skipped']:>7}  {r['errors']:>5}", flush=True)
        tf += r["found"]; ti += r["imported"]; ts_ += r["skipped"]; te += r["errors"]
    print("─" * 52, flush=True)
    print(f"{'TOTAL':<16}  {tf:>5}  {ti:>8}  {ts_:>7}  {te:>5}", flush=True)
    print(f"\nCompleted in {elapsed}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
