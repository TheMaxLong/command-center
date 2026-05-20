#!/usr/bin/env python3.12
"""
COMMAND CENTER — Multi-camera vision watcher v2.

HTTP API on :8181
  GET /status                    → all cameras JSON
  GET /status/<cam_id>           → one camera status
  GET /clip/<cam_id>             → latest clip (video/mp4)
  GET /snap/<cam_id>             → latest snapshot (image/jpeg)
  GET /snap_ann/<cam_id>         → AI-annotated snapshot (image/jpeg)
  GET /events[?camera=&limit=]   → recent events from DB
  GET /trends[?camera=&weeks=]   → full trend report (heatmap + schedule + anomalies + velocity)
  GET /profiles                  → person profiles summary
  GET /thumb/<profile_id>        → profile thumbnail
  GET /profile/<id>/timeline     → cross-camera sightings timeline
  PATCH /profile/<id>/label      → set custom name {"label": "..."}
  GET /intel/briefing[?camera=]  → 24h plain-English briefing
  GET /intel/alerts[?camera=]    → active anomaly alerts
  GET /intel/velocity[?camera=]  → event rate trend
"""

import asyncio, json, os, subprocess, threading, time, yaml
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import ai_engine
import event_db
import profiler
import backfill_tapo
import trend_analyzer
import intel_engine
import query_agent
import intel_feeds
import lpr_engine
import gait_engine
import face_intel
import pattern_engine
import traffic_cam
import camera_adapters
import camera_discover
import entity_resolution
import forward_intel
import notifier
import evidence_export
import vision_tools
import drone_ops

# ── Config ────────────────────────────────────────────────────────
SERVE_PORT  = int(os.environ.get("WATCHER_PORT", "8181"))
CONFIG_FILE = Path(os.environ.get("CAMERAS_CONFIG", "/config/cameras.yaml"))
MEDIA_DIR   = Path(os.environ.get("MEDIA_DIR", "/tmp/cams"))
SCENE_ZONES_FILE = Path(os.environ.get("SCENE_ZONES_FILE", "/data/scene_zones.json"))

PREVIEW_REQ = json.dumps({
    "type": "request", "seq": 1,
    "params": {"preview": {"audio": ["default"], "channels": [0], "resolutions": ["HD"]}, "method": "get"},
})


# ── Per-camera runtime state ──────────────────────────────────────

@dataclass
class CameraState:
    id:              str
    name:            str
    online:          bool            = False
    last_seen:       Optional[float] = None
    events:          int             = 0
    last_mode:       Optional[str]   = None
    last_detections: list            = field(default_factory=list)
    last_profiles:   list            = field(default_factory=list)
    last_summary:    str             = ""
    last_vision:      dict            = field(default_factory=dict)

cameras: dict[str, CameraState] = {}
_watch_modes: dict[str, dict] = {}


def _start_watch_mode(
    cam_id: str,
    minutes: float = 10,
    poll_interval: float = 3,
    cooldown: float = 60,
    capture_duration: int = 8,
) -> dict:
    """Temporarily make a battery camera more attentive after operator request."""
    minutes = max(1.0, min(float(minutes), 60.0))
    mode = {
        "active_until": time.time() + minutes * 60,
        "minutes": minutes,
        "poll_interval": max(1.0, min(float(poll_interval), 30.0)),
        "cooldown": max(0.0, min(float(cooldown), 900.0)),
        "capture_duration": max(3, min(int(capture_duration), 15)),
    }
    _watch_modes[cam_id] = mode
    print(
        f"[watch] {cam_id}: operator watch for {minutes:.0f}m "
        f"poll={mode['poll_interval']}s cooldown={mode['cooldown']}s",
        flush=True,
    )
    return _watch_status(cam_id)


def _watch_status(cam_id: str) -> dict:
    mode = _watch_modes.get(cam_id)
    now = time.time()
    if not mode or mode.get("active_until", 0) <= now:
        _watch_modes.pop(cam_id, None)
        return {"active": False, "remaining_s": 0}
    return {
        "active": True,
        "remaining_s": int(mode["active_until"] - now),
        "poll_interval": mode.get("poll_interval"),
        "cooldown": mode.get("cooldown"),
        "capture_duration": mode.get("capture_duration"),
    }

# ── Exclusion zones: cam_id → list of normalized {x1,y1,x2,y2} rects ─
_exclusion_zones: dict[str, list[dict]] = {}

# ── Known zones: same format, but detections are tagged not dropped ──
_known_zones: dict[str, list[dict]] = {}

# ── Attention zones: operator-defined areas the AI should emphasize ──
_attention_zones: dict[str, list[dict]] = {
    "front_cam": [
        {
            "label": "parking lot",
            "x1": 0.0,
            "y1": 0.0,
            "x2": 0.46,
            "y2": 1.0,
            "priority": "high",
            "note": "Left side of frame; operator focus area.",
        }
    ]
}


def _clean_zone(zone: dict) -> dict:
    def clamp(v, default):
        try:
            return max(0.0, min(1.0, float(v)))
        except Exception:
            return default
    x1 = clamp(zone.get("x1"), 0.0)
    y1 = clamp(zone.get("y1"), 0.0)
    x2 = clamp(zone.get("x2"), 1.0)
    y2 = clamp(zone.get("y2"), 1.0)
    return {
        "label": str(zone.get("label") or "focus").strip()[:48],
        "x1": min(x1, x2),
        "y1": min(y1, y2),
        "x2": max(x1, x2),
        "y2": max(y1, y2),
        "priority": str(zone.get("priority") or "normal").strip()[:16],
        "note": str(zone.get("note") or "").strip()[:160],
    }


def _load_scene_zones() -> None:
    if not SCENE_ZONES_FILE.exists():
        return
    try:
        data = json.loads(SCENE_ZONES_FILE.read_text())
        for cam_id, zones in (data.get("attention_zones") or {}).items():
            if isinstance(zones, list):
                _attention_zones[cam_id] = [_clean_zone(z) for z in zones]
        print(f"[scene] loaded attention zones from {SCENE_ZONES_FILE}", flush=True)
    except Exception as e:
        print(f"[scene] zone load error: {e}", flush=True)


def _save_scene_zones() -> None:
    try:
        SCENE_ZONES_FILE.parent.mkdir(parents=True, exist_ok=True)
        SCENE_ZONES_FILE.write_text(json.dumps({"attention_zones": _attention_zones}, indent=2))
    except Exception as e:
        print(f"[scene] zone save error: {e}", flush=True)


def _tag_attention_zones(cam_id: str, detections: list[dict], snap_path: Optional[str]) -> list[dict]:
    """Tag detections inside operator attention zones."""
    zones = _attention_zones.get(cam_id)
    if not zones or not detections or not snap_path:
        return detections
    try:
        from PIL import Image
        with Image.open(snap_path) as img:
            iw, ih = img.size
    except Exception:
        return detections

    for det in detections:
        cx, cy = det.get("cx", 0), det.get("cy", 0)
        nx, ny = cx / iw, cy / ih
        hit = next(
            (z for z in zones if z["x1"] <= nx <= z["x2"] and z["y1"] <= ny <= z["y2"]),
            None
        )
        if hit:
            det["attention_zone"] = hit.get("label", "focus")
            det["attention_priority"] = hit.get("priority", "normal")
    return detections


def _tag_known_zones(cam_id: str, detections: list[dict], snap_path: Optional[str]) -> list[dict]:
    """Tag detections whose center falls in a known zone. Still recorded, alerts suppressed."""
    zones = _known_zones.get(cam_id)
    if not zones or not detections or not snap_path:
        return detections
    try:
        from PIL import Image
        with Image.open(snap_path) as img:
            iw, ih = img.size
    except Exception:
        return detections

    for det in detections:
        cx, cy = det.get("cx", 0), det.get("cy", 0)
        nx, ny = cx / iw, cy / ih
        hit = next(
            (z for z in zones if z["x1"] <= nx <= z["x2"] and z["y1"] <= ny <= z["y2"]),
            None
        )
        if hit:
            det["known_zone"] = hit.get("label", "known")
    return detections


ARCHIVE_RETENTION_DAYS = int(os.environ.get("ARCHIVE_RETENTION_DAYS", "14"))


def _archive_media(cam_id: str, event_id: int, ts: float, clip_path: Optional[str], snap_path: Optional[str]) -> None:
    """Save timestamped copies of clip + snap to archive dir. Tracks paths in DB."""
    from datetime import datetime, timezone
    dt  = datetime.fromtimestamp(ts, tz=timezone.utc)
    tag = dt.strftime("%Y%m%d_%H%M%S") + f"_{event_id}"
    arc_dir = MEDIA_DIR / cam_id / "archive"
    arc_dir.mkdir(parents=True, exist_ok=True)

    arc_clip = arc_snap = None
    try:
        if clip_path and Path(clip_path).exists():
            dst = arc_dir / f"{tag}.mp4"
            import shutil
            shutil.copy2(clip_path, dst)
            arc_clip = str(dst)
    except Exception as e:
        print(f"[archive] clip copy failed: {e}", flush=True)

    try:
        if snap_path and Path(snap_path).exists():
            dst = arc_dir / f"{tag}.jpg"
            import shutil
            shutil.copy2(snap_path, dst)
            arc_snap = str(dst)
    except Exception as e:
        print(f"[archive] snap copy failed: {e}", flush=True)

    if arc_clip or arc_snap:
        event_db.set_event_archived_paths(event_id, arc_clip, arc_snap)


def _purge_old_media(days: int = ARCHIVE_RETENTION_DAYS, dry_run: bool = False) -> dict:
    """Delete archived files + DB records for non-starred events older than N days."""
    rows   = event_db.get_purgeable_events(days)
    freed  = 0
    ids    = []
    for row in rows:
        for path_key in ("archived_clip", "archived_snap"):
            p = row.get(path_key)
            if p:
                f = Path(p)
                if f.exists():
                    freed += f.stat().st_size
                    if not dry_run:
                        f.unlink(missing_ok=True)
        ids.append(row["id"])
    deleted = 0
    if not dry_run and ids:
        deleted = event_db.purge_events(ids)
    return {"purged_events": len(ids), "deleted_records": deleted, "bytes_freed": freed, "dry_run": dry_run}


def _delete_event_media(rows: list[dict], include_latest: bool = False) -> dict:
    """Delete archived media files for selected events, then report freed space."""
    files_deleted = 0
    bytes_freed = 0
    keys = ["archived_clip", "archived_snap"]
    if include_latest:
        keys += ["clip_path", "snap_path"]

    for row in rows:
        for path_key in keys:
            raw = row.get(path_key)
            if not raw:
                continue
            path = Path(raw)
            try:
                if not path.exists() or not path.is_file():
                    continue
                bytes_freed += path.stat().st_size
                path.unlink()
                files_deleted += 1
            except Exception as e:
                print(f"[storage] delete failed for {path}: {e}", flush=True)
    return {"files_deleted": files_deleted, "bytes_freed": bytes_freed}


def _archive_files() -> list[Path]:
    files: list[Path] = []
    if not MEDIA_DIR.exists():
        return files
    for cam_dir in MEDIA_DIR.iterdir():
        arc = cam_dir / "archive"
        if not arc.is_dir():
            continue
        files.extend(f for f in arc.iterdir() if f.is_file())
    return files


def _referenced_media_paths() -> set[str]:
    refs: set[str] = set()
    for row in event_db.get_all_event_media_rows():
        for key in ("archived_clip", "archived_snap"):
            raw = row.get(key)
            if raw:
                refs.add(str(Path(raw)))
    return refs


def _orphaned_archive_media(dry_run: bool = True) -> dict:
    """Find or delete archive files that no longer have an event DB row."""
    refs = _referenced_media_paths()
    orphaned = [f for f in _archive_files() if str(f) not in refs]
    bytes_found = 0
    files_deleted = 0
    sample = []
    for f in orphaned:
        try:
            size = f.stat().st_size
            bytes_found += size
            if len(sample) < 8:
                sample.append(str(f))
            if not dry_run:
                f.unlink()
                files_deleted += 1
        except Exception as e:
            print(f"[storage] orphan cleanup failed for {f}: {e}", flush=True)
    return {
        "orphan_files": len(orphaned),
        "orphan_bytes": bytes_found,
        "files_deleted": files_deleted,
        "bytes_freed": bytes_found if not dry_run else 0,
        "dry_run": dry_run,
        "sample": sample,
    }


def _filter_exclusions(cam_id: str, detections: list[dict], snap_path: Optional[str]) -> list[dict]:
    """Drop detections whose center falls inside a configured exclusion zone."""
    zones = _exclusion_zones.get(cam_id)
    if not zones or not detections or not snap_path:
        return detections
    try:
        from PIL import Image
        with Image.open(snap_path) as img:
            iw, ih = img.size
    except Exception:
        return detections

    kept = []
    for det in detections:
        cx, cy = det.get("cx", 0), det.get("cy", 0)
        nx, ny = cx / iw, cy / ih
        hit = next(
            (z for z in zones if z["x1"] <= nx <= z["x2"] and z["y1"] <= ny <= z["y2"]),
            None
        )
        if hit:
            print(f"[exclusion] {cam_id}: dropped {det['class']} ({det.get('confidence')}) in zone '{hit.get('label','?')}'", flush=True)
        else:
            kept.append(det)
    return kept


# ── Media helpers ─────────────────────────────────────────────────

def media_path(cam_id: str, kind: str) -> Path:
    d = MEDIA_DIR / cam_id
    d.mkdir(parents=True, exist_ok=True)
    if kind == "clip":
        ext = "mp4"
    elif kind == "audio":
        ext = "wav"
    else:
        ext = "jpg"  # snap, etc.
    return d / f"{kind}.{ext}"


def _ffmpeg_from_ts(ts_data: bytes, cam_id: str, capture_s: int) -> tuple[bool, bool]:
    clip = media_path(cam_id, "clip")
    snap = media_path(cam_id, "snap")
    clip_ok = snap_ok = False
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "mpegts", "-i", "pipe:0",
             "-c:v", "copy", "-an", "-movflags", "+faststart",
             "-t", str(capture_s), str(clip)],
            input=ts_data, capture_output=True, timeout=30,
        )
        clip_ok = clip.exists() and clip.stat().st_size > 10_000
    except Exception:
        pass
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "mpegts", "-i", "pipe:0",
             "-vframes", "1", "-q:v", "2", "-f", "image2", str(snap)],
            input=ts_data, capture_output=True, timeout=20,
        )
        snap_ok = snap.exists() and snap.stat().st_size > 0
    except Exception:
        pass
    return clip_ok, snap_ok


def _ffmpeg_snap_rtsp(rtsp_url: str, cam_id: str) -> bool:
    snap = media_path(cam_id, "snap")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-rtsp_transport", "tcp", "-i", rtsp_url,
             "-vframes", "1", "-q:v", "2", "-f", "image2", str(snap)],
            capture_output=True, timeout=15,
        )
        return snap.exists() and snap.stat().st_size > 0
    except Exception:
        return False


def _ffmpeg_clip_rtsp(rtsp_url: str, cam_id: str, duration: int) -> bool:
    clip = media_path(cam_id, "clip")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-rtsp_transport", "tcp", "-i", rtsp_url,
             "-t", str(duration), "-c:v", "copy", "-an",
             "-movflags", "+faststart", str(clip)],
            capture_output=True, timeout=duration + 15,
        )
        return clip.exists() and clip.stat().st_size > 10_000
    except Exception:
        return False


# ── AI + DB integration ───────────────────────────────────────────

def _run_ai_and_store(
    cam_id: str,
    event_ts: float,
    clip_path: Optional[str],
    snap_path: Optional[str],
    audio_enabled: bool = False,
    cam_cfg: Optional[dict] = None,
) -> None:
    """Run AI on snapshot + optional audio, persist event + detections. Called in a daemon thread."""
    detections: list[dict] = []

    # Extract and classify audio if enabled
    audio_wav = None
    if audio_enabled and clip_path:
        audio_wav = _extract_audio_wav(clip_path, cam_id)
        _process_audio_events(cam_id, event_ts, audio_wav, audio_enabled)
    if snap_path and Path(snap_path).exists():
        detections = ai_engine.detect(snap_path)
        detections = _filter_exclusions(cam_id, detections, snap_path)
        detections = _tag_known_zones(cam_id, detections, snap_path)
        detections = _tag_attention_zones(cam_id, detections, snap_path)

    # ── Person tracking: Kalman filter + Hungarian assignment (ByteTrack)
    if detections:
        try:
            import tracker_kalman
            tracker    = tracker_kalman.get_tracker(cam_id)
            detections = tracker.update(detections, event_ts)
        except Exception as te:
            # Fallback to legacy IoU tracker if Kalman fails
            tracker    = ai_engine.get_tracker(cam_id)
            detections = tracker.update(detections, event_ts)

    # ── Pose + gait landmarks: posture tags, skeleton points, gait signature
    if snap_path and detections and any(d.get("class") == "person" for d in detections):
        try:
            detections = gait_engine.process_frame(snap_path, detections, cam_id, event_ts)
        except Exception as ge:
            print(f"[pose] error: {ge}", flush=True)

    detections = vision_tools.enrich_detections(
        cam_id,
        detections,
        snap_path,
        _attention_zones.get(cam_id, []),
        _known_zones.get(cam_id, []),
    )
    vision_summary = vision_tools.scene_metrics(
        cam_id,
        detections,
        _attention_zones.get(cam_id, []),
    )

    # ── License plate recognition on vehicle detections
    if snap_path and detections:
        vehicle_dets = [d for d in detections if d.get("class") in ("car","truck","bus","motorcycle")]
        if vehicle_dets:
            try:
                lpr_engine.process_snapshot(snap_path, vehicle_dets, cam_id, event_ts)
            except Exception as le:
                print(f"[lpr] error: {le}", flush=True)

    event_id = event_db.insert_event(cam_id, event_ts, clip_path, snap_path)
    if detections:
        event_db.add_detections(event_id, detections)

    # ── Archive timestamped copies of clip + snap
    _archive_media(cam_id, event_id, event_ts, clip_path, snap_path)

    # ── FreeMoCap auto-extract hook (doorbell motion clip only)
    if os.getenv("FREEMOCAP_AUTO_EXTRACT") == "1" and cam_id == "doorbell" and clip_path:
        try:
            import freemocap_worker
            worker = freemocap_worker.get_worker()
            worker.ingest_doorbell_clip(
                event_id=str(event_id),
                clip_path=Path(clip_path),
                metadata={
                    "camera": cam_id,
                    "timestamp": event_ts,
                    "detections_count": len(detections),
                    "has_person": any(d.get("class") == "person" for d in detections),
                }
            )
            print(f"[mocap] queued doorbell clip {event_id} for FreeMoCap processing", flush=True)
        except Exception as e:
            print(f"[mocap] doorbell ingest failed: {e}", flush=True)

    if snap_path and detections:
        ai_engine.annotate(
            snap_path,
            detections,
            attention_zones=_attention_zones.get(cam_id, []),
            known_zones=_known_zones.get(cam_id, []),
        )

    # Person profiling
    profile_ids: list[int] = []
    profile_objs: list[dict] = []
    if snap_path and detections:
        crops = ai_engine.extract_crops(snap_path, detections)
        profile_ids = profiler.match_or_create(cam_id, event_ts, crops, event_id)
        profile_objs = [
            {
                "id":        pid,
                "label":     profiler.get_profile_label(pid),
                "is_regular": profiler.get_profile_label(pid).startswith("REGULAR"),
            }
            for pid in profile_ids
        ]

    # ── Face intelligence — compare against FBI + POI database
    if snap_path and detections:
        for det in detections:
            if det.get("class") == "person" and det.get("track_id"):
                try:
                    matches = face_intel.compare_detection(snap_path, det, cam_id, event_ts)
                    if matches:
                        top = matches[0]
                        det["face_match"] = top
                        if top["confidence"] in ("HIGH", "MEDIUM"):
                            alert_msg = (f"FACE INTEL HIT: {top['name']} "
                                         f"({top['source']}) conf={top['confidence']} on {cam_id}")
                            event_db.insert_alert("face_intel", "critical", alert_msg, cam_id, event_ts)
                            notifier.notify_critical("face_intel", alert_msg, cam_id)
                            print(f"[face_intel] ⚠ {alert_msg}", flush=True)
                except Exception as fe:
                    print(f"[face_intel] error: {fe}", flush=True)

    # Check for stranger alerts — skip if detection came from a known zone
    known_zone_pids = {
        profile_ids[i]
        for i, d in enumerate(detections[:len(profile_ids)])
        if d.get("known_zone")
    }
    for pid in profile_ids:
        if pid in known_zone_pids:
            label = profiler.get_profile_label(pid)
            if not label.startswith("NEIGHBOR"):
                profiler.set_label(pid, label.replace("REGULAR", "NEIGHBOR").replace("UNKNOWN", "NEIGHBOR"))
            continue
        if intel_engine.stranger_alert(pid, event_ts, cam_id):
            msg = f"Unknown person on {cam_id.upper()} during off-peak hours"
            event_db.insert_alert("stranger", "warn", msg, cam_id, event_ts)
            notifier.notify("stranger", "medium", msg, cam_id)
            print(f"[intel] ALERT: {msg}", flush=True)

    # ── Entity resolution: fuse face + gait + appearance into single identity
    if profile_ids:
        try:
            resolver = entity_resolution.get_resolver()
            for i, pid in enumerate(profile_ids):
                d = detections[i] if i < len(detections) else {}
                fv = (d.get("face_match") or {}).get("vec") or d.get("face_vec")
                gv = d.get("gait_vec")
                ap = d.get("appearance")
                resolver.observe(profile_id=str(pid), camera=cam_id, ts=event_ts,
                                 face_vec=fv, gait_vec=gv, appearance=ap)
        except Exception as ee:
            print(f"[entity_res] error: {ee}", flush=True)

    # ── Pattern-of-life threat scoring for each confirmed person
    if profile_ids:
        try:
            eng = pattern_engine.get_engine()
            eng.build()
            for i, pid in enumerate(profile_ids):
                gait_conf = 0.0
                face_conf = 0.0
                if i < len(detections):
                    d = detections[i]
                    gait_conf = d.get("gait_conf") or 0.0
                    face_conf = (d.get("face_match") or {}).get("similarity") or 0.0
                score = eng.score_appearance(pid, event_ts, cam_id, gait_conf, face_conf)
                if score["level"] in ("RED", "ORANGE"):
                    alert_msg = (f"THREAT SCORE {score['level']}: "
                                 f"Profile-{pid:03d} on {cam_id} · "
                                 f"{' · '.join(score['reasons'][:2])}")
                    event_db.insert_alert("threat_score", score["level"].lower(),
                                          alert_msg, cam_id, event_ts)
                    sev = "critical" if score["level"] == "RED" else "high"
                    notifier.notify("threat_score", sev, alert_msg, cam_id)
        except Exception as pe:
            print(f"[pattern] scoring error: {pe}", flush=True)

    # Build scene summary
    summary = intel_engine.scene_summary(detections, profile_objs, cam_id)

    state = cameras.get(cam_id)
    if state:
        state.last_detections = detections
        state.last_profiles   = profile_objs
        state.last_summary    = summary
        state.last_vision     = vision_summary

    if detections:
        ts_str = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts_str}] [{cam_id}] {summary}", flush=True)


def _extract_audio_wav(clip_path: Optional[str], cam_id: str) -> Optional[Path]:
    """Extract audio from MP4 clip as WAV. Returns Path to WAV or None on failure."""
    if not clip_path or not Path(clip_path).exists():
        return None
    try:
        wav = media_path(cam_id, "audio")
        subprocess.run(
            ["ffmpeg", "-y", "-i", clip_path,
             "-q:a", "9", "-ac", "1", "-ar", "16000",
             "-f", "wav", str(wav)],
            capture_output=True, timeout=30,
        )
        if wav.exists() and wav.stat().st_size > 1000:
            return wav
    except Exception as e:
        print(f"[audio] extraction failed for {cam_id}: {e}", flush=True)
    return None


def _process_audio_events(
    cam_id: str,
    event_ts: float,
    audio_wav_path: Optional[Path],
    config_enabled: bool,
) -> None:
    """Classify audio events and insert into DB (gated by config flag)."""
    if not config_enabled or not audio_wav_path or not audio_wav_path.exists():
        return
    try:
        import audio_engine
        events = audio_engine.classify_audio(audio_wav_path)
        for evt in events:
            event_db.insert_audio_event(
                camera_id=cam_id,
                ts=event_ts + evt["start_s"],  # adjust to actual event time
                class_name=evt["class_name"],
                confidence=evt["confidence"],
                audio_clip_path=str(audio_wav_path),
            )
        if events:
            print(f"[{cam_id}] logged {len(events)} audio events", flush=True)
    except Exception as e:
        print(f"[audio] classification failed for {cam_id}: {e}", flush=True)


def _fire_ai(cam_id: str, event_ts: float, clip_ok: bool, snap_ok: bool, cam_cfg: Optional[dict] = None) -> None:
    clip_p = str(media_path(cam_id, "clip")) if clip_ok else None
    snap_p = str(media_path(cam_id, "snap")) if snap_ok else None
    audio_enabled = cam_cfg.get("audio_events", False) if cam_cfg else False
    threading.Thread(
        target=_run_ai_and_store,
        args=(cam_id, event_ts, clip_p, snap_p, audio_enabled, cam_cfg),
        daemon=True,
    ).start()


# ── Tapo camera watcher ───────────────────────────────────────────

async def _tapo_capture(cam_cfg: dict, cam_id: str) -> tuple[bool, bool]:
    from pytapo import HttpMediaSession
    from pytapo.const import EncryptionMethod

    ip  = cam_cfg["ip"]
    port = int(cam_cfg.get("port", 8800))
    pwd  = cam_cfg["password"]
    watch = _watch_status(cam_id)
    dur  = int(watch.get("capture_duration") or cam_cfg.get("capture_duration", 9))

    session = HttpMediaSession(
        ip=ip, cloud_password=pwd, super_secret_key="",
        encryptionMethod=EncryptionMethod.SHA256, port=port, window_size=50,
    )
    ts_buf = bytearray()
    try:
        await asyncio.wait_for(session.start(), timeout=8)
        deadline = time.monotonic() + dur
        async for resp in session.transceive(PREVIEW_REQ, no_data_timeout=4.0):
            if resp.mimetype == "video/mp2t" and isinstance(resp.plaintext, bytes):
                ts_buf.extend(resp.plaintext)
            if time.monotonic() >= deadline:
                break
    except Exception as e:
        print(f"  [{cam_id}] stream: {e}", flush=True)
    finally:
        try:
            await session.close()
        except Exception:
            pass

    if len(ts_buf) < 4096:
        print(f"  [{cam_id}] only {len(ts_buf)}b — skipping", flush=True)
        return False, False

    return _ffmpeg_from_ts(bytes(ts_buf), cam_id, dur)


async def tapo_poll_loop(cam_cfg: dict) -> None:
    cam_id     = cam_cfg["id"]
    ip         = cam_cfg["ip"]
    port       = int(cam_cfg.get("port", 8800))
    base_poll_s = float(cam_cfg.get("poll_interval", 3))
    base_cooldown_s = float(cam_cfg.get("cooldown", 300))
    # Require this many consecutive open polls before firing.
    # Wind/trees cause single-poll blips; a person walking up holds the port
    # open for multiple polls in a row. Default 2 = one confirmation poll.
    confirm_needed = int(cam_cfg.get("motion_confirm", 2))

    was_online     = False
    capturing      = False
    last_cap       = 0.0
    confirm_streak = 0          # consecutive polls where port was open
    state          = cameras[cam_id]

    print(f"[{cam_id}] Tapo watcher — {ip}:{port} every {base_poll_s}s "
          f"cooldown={base_cooldown_s}s confirm={confirm_needed}", flush=True)

    while True:
        watch = _watch_status(cam_id)
        poll_s = float(watch.get("poll_interval") or base_poll_s)
        cooldown_s = float(watch.get("cooldown") or base_cooldown_s)
        try:
            _, w = await asyncio.wait_for(asyncio.open_connection(ip, port), timeout=3)
            w.close()
            try:
                await w.wait_closed()
            except Exception:
                pass

            # Port is open — increment confirmation streak
            confirm_streak += 1

            in_cooldown = (time.time() - last_cap) < cooldown_s
            # Fire only once we've seen enough consecutive opens AND we're not
            # already capturing or in cooldown
            if confirm_streak >= confirm_needed and not was_online and not capturing and not in_cooldown:
                was_online      = True
                capturing       = True
                confirm_streak  = 0
                state.online    = True
                state.last_seen = time.time()
                state.events   += 1
                ts_str = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts_str}] [{cam_id}] MOTION #{state.events} "
                      f"(confirmed {confirm_needed} polls)", flush=True)

                async def _capture(cfg=cam_cfg, cid=cam_id):
                    nonlocal capturing, was_online, last_cap
                    try:
                        event_ts = time.time()
                        clip_ok, snap_ok = await _tapo_capture(cfg, cid)
                        if clip_ok or snap_ok:
                            state.last_mode = "clip" if clip_ok else "snap"
                            _fire_ai(cid, event_ts, clip_ok, snap_ok, cam_cfg=cfg)
                        last_cap = time.time()
                    finally:
                        capturing    = False
                        was_online   = False
                        state.online = False

                asyncio.create_task(_capture())

        except Exception:
            # Port closed — reset streak, clear online flag
            confirm_streak = 0
            if was_online and not capturing:
                state.online = False
            if not capturing:
                was_online = False

        await asyncio.sleep(poll_s)


# ── go2rtc snapshot-based AI watcher ─────────────────────────────

async def go2rtc_poll_loop(cam_cfg: dict) -> None:
    import urllib.request, urllib.error

    cam_id   = cam_cfg["id"]
    base_url = cam_cfg.get("go2rtc_url", "http://go2rtc:1984").rstrip("/")
    src_name = cam_cfg.get("source_name", cam_id)
    interval = float(cam_cfg.get("ai_interval", 10))
    state    = cameras[cam_id]
    snap     = media_path(cam_id, "snap")

    print(f"[{cam_id}] go2rtc AI watcher — {base_url}/api/frame.jpeg?src={src_name} every {interval}s", flush=True)

    while True:
        await asyncio.sleep(interval)
        try:
            url = f"{base_url}/api/frame.jpeg?src={src_name}"
            with urllib.request.urlopen(url, timeout=5) as r:
                data = r.read()
            if len(data) < 512:
                continue
            snap.parent.mkdir(parents=True, exist_ok=True)
            snap.write_bytes(data)
            state.online    = True
            state.last_seen = time.time()

            detections  = ai_engine.detect(snap)
            detections = _filter_exclusions(cam_id, detections, str(snap))
            detections = _tag_known_zones(cam_id, detections, str(snap))
            detections = _tag_attention_zones(cam_id, detections, str(snap))
            if detections:
                try:
                    import tracker_kalman
                    detections = tracker_kalman.get_tracker(cam_id).update(detections, state.last_seen)
                except Exception:
                    detections = ai_engine.get_tracker(cam_id).update(detections, state.last_seen)
            if detections and any(d.get("class") == "person" for d in detections):
                try:
                    detections = gait_engine.process_frame(str(snap), detections, cam_id, state.last_seen)
                except Exception as ge:
                    print(f"[pose] {cam_id} error: {ge}", flush=True)
            detections = vision_tools.enrich_detections(
                cam_id,
                detections,
                str(snap),
                _attention_zones.get(cam_id, []),
                _known_zones.get(cam_id, []),
            )
            if detections:
                ai_engine.annotate(
                    snap,
                    detections,
                    attention_zones=_attention_zones.get(cam_id, []),
                    known_zones=_known_zones.get(cam_id, []),
                )
            crops       = ai_engine.extract_crops(snap, detections)
            profile_ids = profiler.match_or_create(cam_id, state.last_seen, crops)
            profile_objs = [
                {"id": pid, "label": profiler.get_profile_label(pid),
                 "is_regular": profiler.get_profile_label(pid).startswith("REGULAR")}
                for pid in profile_ids
            ]

            state.last_detections = detections
            state.last_profiles   = profile_objs
            state.last_summary    = intel_engine.scene_summary(detections, profile_objs, cam_id)
            state.last_vision     = vision_tools.scene_metrics(cam_id, detections, _attention_zones.get(cam_id, []))

            if detections:
                ts_str = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts_str}] [{cam_id}] {state.last_summary}", flush=True)

        except urllib.error.URLError:
            state.online = False
        except Exception as e:
            print(f"[{cam_id}] go2rtc AI error: {e}", flush=True)
            state.online = False


# ── Universal adapter watcher (any registered vendor) ─────────────

async def adapter_poll_loop(cam_cfg: dict) -> None:
    """Snapshot-poll loop driven by the camera_adapters registry.

    Works for ANY vendor with a registered adapter (hikvision, reolink,
    amcrest, onvif, mjpeg, http_snap, wyze, usb, bluetooth, etc.).
    Snapshot interval is set by `ai_interval` in cameras.yaml (default 8s).
    """
    cam_id   = cam_cfg["id"]
    interval = float(cam_cfg.get("ai_interval", 8))
    state    = cameras[cam_id]
    snap     = media_path(cam_id, "snap")

    adapter = camera_adapters.build_adapter(cam_cfg)
    if adapter is None:
        print(f"[{cam_id}] No adapter for type={cam_cfg.get('type')} — skipping", flush=True)
        return

    print(f"[{cam_id}] {adapter.vendor} adapter @ {adapter.ip}:{adapter.port} every {interval}s "
          f"caps={adapter.capabilities()}", flush=True)

    loop = asyncio.get_event_loop()
    fail_streak = 0
    while True:
        await asyncio.sleep(interval)
        try:
            data = await loop.run_in_executor(None, adapter.snapshot)
            if not data or len(data) < 512:
                fail_streak += 1
                state.online = False
                if fail_streak == 1:
                    print(f"[{cam_id}] {adapter.vendor} snapshot empty/short", flush=True)
                continue
            fail_streak = 0
            snap.parent.mkdir(parents=True, exist_ok=True)
            snap.write_bytes(data)
            state.online    = True
            state.last_seen = time.time()
            event_ts        = state.last_seen

            # Run the entire AI pipeline in a worker thread so the event
            # loop is never blocked by detection/annotation/face/gait/entity calls
            await loop.run_in_executor(None, _run_ai_and_store,
                                       cam_id, event_ts, None, str(snap))

        except Exception as e:
            fail_streak += 1
            state.online = False
            if fail_streak <= 2:
                print(f"[{cam_id}] adapter error: {e}", flush=True)


# ── RTSP camera watcher ───────────────────────────────────────────

async def rtsp_poll_loop(cam_cfg: dict) -> None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        print(f"[{cam_cfg['id']}] RTSP watcher needs opencv-python-headless and numpy.", flush=True)
        return

    cam_id    = cam_cfg["id"]
    rtsp_url  = cam_cfg["rtsp_url"]
    poll_s    = float(cam_cfg.get("poll_interval", 5))
    threshold = float(cam_cfg.get("motion_threshold", 0.02))
    dur       = int(cam_cfg.get("capture_duration", 9))
    state     = cameras[cam_id]

    print(f"[{cam_id}] RTSP watcher — {rtsp_url}", flush=True)

    prev_gray = None
    capturing = False

    while True:
        try:
            cap = cv2.VideoCapture(rtsp_url)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                state.online = False
                await asyncio.sleep(poll_s)
                continue

            state.online = True
            gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)

            if prev_gray is not None and not capturing:
                diff  = cv2.absdiff(prev_gray, gray)
                score = float(np.count_nonzero(diff > 25)) / diff.size
                if score > threshold:
                    capturing        = True
                    state.events    += 1
                    state.last_seen  = time.time()
                    event_ts         = time.time()
                    ts_str = datetime.now().strftime("%H:%M:%S")
                    print(f"[{ts_str}] [{cam_id}] MOTION #{state.events} (score={score:.3f})", flush=True)

                    def _rtsp_capture(url=rtsp_url, cid=cam_id, ets=event_ts, d=dur):
                        nonlocal capturing
                        try:
                            snap_url = cam_cfg.get("snapshot_url")
                            if snap_url:
                                import urllib.request
                                urllib.request.urlretrieve(snap_url, str(media_path(cid, "snap")))
                                snap_ok = media_path(cid, "snap").stat().st_size > 0
                            else:
                                snap_ok = _ffmpeg_snap_rtsp(url, cid)
                            clip_ok = _ffmpeg_clip_rtsp(url, cid, d)
                            cameras[cid].last_mode = "clip" if clip_ok else "snap"
                            _fire_ai(cid, ets, clip_ok, snap_ok)
                        finally:
                            nonlocal capturing
                            capturing = False
                            cameras[cid].online = False

                    threading.Thread(target=_rtsp_capture, daemon=True).start()

            prev_gray = gray

        except Exception as e:
            print(f"[{cam_id}] RTSP error: {e}", flush=True)
            state.online = False

        await asyncio.sleep(poll_s)


# ── Field Scan helpers (phone app API) ───────────────────────────

def _parse_upload(handler) -> Optional[bytes]:
    ct     = handler.headers.get("Content-Type", "")
    length = int(handler.headers.get("Content-Length", 0))
    body   = handler.rfile.read(length)
    if "boundary=" not in ct:
        return body
    boundary = ct.split("boundary=")[-1].strip().encode()
    for part in body.split(b"--" + boundary):
        if b'name="image"' in part or b"name='image'" in part:
            idx = part.find(b"\r\n\r\n")
            if idx != -1:
                return part[idx + 4:].rstrip(b"\r\n-")
    return None


def _field_scan_plate(image_bytes: bytes) -> dict:
    try:
        import cv2
        import numpy as np
        arr  = np.frombuffer(image_bytes, np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"plate": "", "confidence": 0.0}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        best, best_conf = "", 0.0

        def _try_ocr(crop):
            nonlocal best, best_conf
            plate, conf = lpr_engine._ocr_region_opencv(crop)
            if conf > best_conf and lpr_engine._is_valid_plate(plate):
                best, best_conf = lpr_engine._normalize_plate(plate), conf
            plate, conf = lpr_engine._ocr_region_easyocr(crop)
            if conf > best_conf and lpr_engine._is_valid_plate(plate):
                best, best_conf = lpr_engine._normalize_plate(plate), conf

        # Try contour-detected candidates first
        candidates = lpr_engine._find_plate_candidates(gray)
        for (x, y, w, h) in candidates:
            _try_ocr(gray[y:y+h, x:x+w])

        # Fallback: phone shot where plate fills the frame — OCR the full image
        # and center-weighted crops (phone photos don't need contour detection)
        if not best:
            _try_ocr(gray)
            h, w = gray.shape
            # center 60% crop — plate usually centered in a deliberate phone shot
            cy, cx = h // 2, w // 2
            _try_ocr(gray[cy - h//3 : cy + h//3, cx - w//3 : cx + w//3])

        return {"plate": best, "confidence": round(best_conf, 3)}
    except Exception as e:
        return {"plate": "", "confidence": 0.0, "error": str(e)}


def _field_scan_face(image_bytes: bytes) -> dict:
    try:
        import cv2
        import numpy as np
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"faces_detected": 0, "fbi_match": False, "match_name": None, "match_details": None}
        faces = face_intel.detect_faces_in_image(img)
        if not faces:
            return {"faces_detected": 0, "fbi_match": False, "match_name": None, "match_details": None}
        engine = face_intel.get_engine()
        all_matches: list[dict] = []
        for (x, y, w, h) in faces:
            crop = img[y:y+h, x:x+w]
            all_matches.extend(engine.compare_face(crop, camera_id="field_scan"))
        all_matches.sort(key=lambda m: -m["similarity"])
        if all_matches:
            top = all_matches[0]
            return {
                "faces_detected": len(faces),
                "fbi_match":      True,
                "match_name":     top.get("name"),
                "match_details":  top.get("description") or top.get("subjects"),
                "similarity":     top.get("similarity"),
                "confidence":     top.get("confidence"),
                "source":         top.get("source"),
                "field_office":   top.get("field_office"),
                "photo_url":      top.get("photo_url"),
                "reward":         top.get("reward"),
            }
        return {"faces_detected": len(faces), "fbi_match": False, "match_name": None, "match_details": None}
    except Exception as e:
        return {"faces_detected": 0, "fbi_match": False, "match_name": None, "match_details": None, "error": str(e)}


# ── HTTP server ───────────────────────────────────────────────────

_PALM_API_TOKEN = os.environ.get("PALM_API_TOKEN", "")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a) -> None:
        pass

    def _auth_ok(self) -> bool:
        """Return True if request is authorised. Localhost (dashboard proxy) always passes."""
        if not _PALM_API_TOKEN:
            return True                          # auth disabled — dev mode
        client_ip = self.client_address[0]
        if client_ip in ("127.0.0.1", "::1"):
            return True                          # internal proxy — always trusted
        sent = self.headers.get("X-Palm-Token", "")
        return sent == _PALM_API_TOKEN

    def _unauthorized(self) -> None:
        body = json.dumps({
            "error": "unauthorized",
            "hint":  "Include header X-Palm-Token: <your PALM_API_TOKEN>",
        }).encode()
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("WWW-Authenticate", 'Bearer realm="COMMAND CENTER"')
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if not self._auth_ok():
            self._unauthorized(); return
        parsed = urlparse(self.path)
        parts  = parsed.path.strip("/").split("/")
        qs     = parse_qs(parsed.query)

        def qp(key: str) -> Optional[str]:
            return qs[key][0] if key in qs else None

        p = parsed.path.rstrip("/")

        # /status  or  /status/<cam_id>
        if p == "/status":
            self._json({cid: self._cam_status(cid) for cid in cameras})

        elif parts[0] == "status" and len(parts) == 2:
            cam_id = parts[1]
            if cam_id not in cameras:
                self._not_found(); return
            self._json(self._cam_status(cam_id))

        # /clip/<cam_id>  /snap/<cam_id>  /snap_ann/<cam_id>
        elif parts[0] in ("clip", "snap", "snap_ann") and len(parts) == 2:
            cam_id = parts[1]
            kind   = parts[0]
            if cam_id not in cameras:
                self._not_found(); return
            if kind == "clip":
                mime = "video/mp4"
                path = media_path(cam_id, "clip")
            elif kind == "snap_ann":
                mime = "image/jpeg"
                path = media_path(cam_id, "snap").parent / "snap_ann.jpg"
            else:
                mime = "image/jpeg"
                path = media_path(cam_id, "snap")
            self._serve_file(path, mime)

        # /profiles
        elif p == "/profiles":
            self._json(profiler.profiles_summary())

        # /thumb/<profile_id>
        elif parts[0] == "thumb" and len(parts) == 2:
            try:
                pid   = int(parts[1])
                thumb = event_db.get_profile_thumb(pid)
            except Exception:
                thumb = None
            if not thumb:
                self._not_found(); return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(thumb)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(thumb)

        # /profile/<id>/timeline
        elif parts[0] == "profile" and len(parts) == 3 and parts[2] == "timeline":
            try:
                pid = int(parts[1])
            except ValueError:
                self._not_found(); return
            limit = int(qp("limit") or 20)
            self._json(intel_engine.cross_camera_timeline(pid, limit))

        # /events[?camera=&limit=]
        elif p == "/events":
            camera = qp("camera")
            limit  = int(qp("limit") or 50)
            self._json(event_db.get_recent_events(camera, limit))

        # /audio-events[?camera=&limit=&since=]  — YAMNet sound event log
        elif p == "/audio-events":
            camera = qp("camera")
            limit  = int(qp("limit") or 50)
            since_str = qp("since")
            since = None
            if since_str:
                try:
                    from datetime import datetime, timezone
                    since = datetime.fromisoformat(since_str.replace("Z", "+00:00")).timestamp()
                except Exception:
                    pass
            events = event_db.get_audio_events(camera, since, limit)
            self._json({"events": events, "count": len(events)})

        # /trends[?camera=&weeks=]  — now uses full trend_analyzer
        elif p == "/trends":
            camera = qp("camera")
            weeks  = int(qp("weeks") or 5)
            self._json(trend_analyzer.analyze(camera, weeks))

        # /intel/briefing[?camera=]
        elif p == "/intel/briefing":
            camera = qp("camera")
            self._json(intel_engine.daily_briefing(camera))

        # /intel/morning-brief[?force=1]  — local LLM (Ollama) narrative
        elif p == "/intel/morning-brief":
            try:
                import llm_brief as _llm
                force = qp("force") in ("1", "true", "yes")
                self._json(_llm.generate_morning_brief(force=force))
            except ImportError as e:
                self._json({"error": f"llm_brief unavailable: {e}"})

        # /intel/anomaly-baseline[?weeks=4&camera=]  — statistical drift heatmap
        elif p == "/intel/anomaly-baseline":
            try:
                import anomaly_baseline as _ab
                weeks = int(qp("weeks") or 4)
                camera = qp("camera")
                self._json(_ab.baseline_grid(weeks=weeks, camera_id=camera))
            except (ImportError, ValueError) as e:
                self._json({"error": f"anomaly_baseline unavailable: {e}"})

        # /intel/anomaly-cell?dow=N&hour=N[&days=28&camera=&limit=30]
        elif p == "/intel/anomaly-cell":
            try:
                import anomaly_baseline as _ab
                dow = int(qp("dow") or 0)
                hour = int(qp("hour") or 0)
                days = int(qp("days") or 28)
                camera = qp("camera")
                limit = int(qp("limit") or 30)
                self._json({"events": _ab.cell_events(dow, hour, days=days, camera_id=camera, limit=limit)})
            except (ImportError, ValueError) as e:
                self._json({"error": f"anomaly_baseline unavailable: {e}"})

        # /intel/alerts[?camera=]
        elif p == "/intel/alerts":
            camera = qp("camera")
            self._json(intel_engine.active_alerts(camera))

        # /intel/velocity[?camera=]
        elif p == "/intel/velocity":
            camera = qp("camera")
            self._json(trend_analyzer.velocity(camera))

        # /feeds              → all intel feeds combined
        elif p == "/feeds":
            self._json(intel_feeds.get_all_feeds())

        # /feeds/earthquakes  → USGS seismic data
        elif p == "/feeds/earthquakes":
            self._json({"items": intel_feeds.get_earthquakes(),
                        "count": len(intel_feeds.get_earthquakes())})

        # /feeds/weather      → NWS alerts
        elif p == "/feeds/weather":
            self._json({"items": intel_feeds.get_weather_alerts(),
                        "count": len(intel_feeds.get_weather_alerts())})

        # /feeds/fire         → CAL FIRE incidents
        elif p == "/feeds/fire":
            self._json({"items": intel_feeds.get_fire_incidents(),
                        "count": len(intel_feeds.get_fire_incidents())})

        # /feeds/crime        → Citizen app local incidents
        elif p == "/feeds/crime":
            self._json({"items": intel_feeds.get_citizen_incidents(),
                        "count": len(intel_feeds.get_citizen_incidents())})

        # /feeds/aqi          → PurpleAir air quality (median + max AQI in ~5km bbox)
        # Requires PURPLEAIR_API_KEY env var; returns {status: "unconfigured"} if missing.
        elif p == "/feeds/aqi":
            self._json(intel_feeds.get_aqi_summary())

        # /feeds/lightning    → Blitzortung lightning strikes (real-time MQTT)
        # Subscribes lazily on first request. Filtered to LIGHTNING_RADIUS_KM (default 50km).
        elif p == "/feeds/lightning":
            strikes = intel_feeds.get_lightning_recent()
            self._json({
                "items":   strikes,
                "summary": intel_feeds.lightning_summary(),
                "count":   len(strikes),
            })

        # /feeds/lightning-global → ALL strikes worldwide (last 10 min default)
        # For the dashboard globe widget. Unfiltered by distance.
        elif p == "/feeds/lightning-global":
            try:
                window = int(qp("seconds") or 600)
            except ValueError:
                window = 600
            strikes = intel_feeds.get_lightning_global(max_age_seconds=window)
            self._json({"items": strikes, "count": len(strikes), "window_s": window})

        # /feeds/plates       → LPR plate log
        elif p == "/feeds/plates":
            camera = qp("camera")
            limit  = int(qp("limit") or 50)
            self._json({
                "plates":  lpr_engine.get_plate_log(camera, limit),
                "unique":  lpr_engine.get_unique_plates(24),
            })

        # /feeds/briefing     → plain-English threat summary
        elif p == "/feeds/briefing":
            self._json({"briefing": intel_feeds.generate_briefing()})

        # /feeds/gait         → gait biometric profiles
        elif p == "/feeds/gait":
            profiles = gait_engine.get_gait_profiles()
            self._json({"profiles": profiles, "count": len(profiles)})

        # /intel/patterns     → pattern-of-life summary for all entities
        elif p == "/intel/patterns":
            patterns = pattern_engine.get_all_patterns()
            self._json({"patterns": patterns, "count": len(patterns)})

        # /intel/predictions  → arrival predictions for known regulars
        elif p == "/intel/predictions":
            self._json({"predictions": pattern_engine.get_predictions()})

        # /intel/graph        → entity relationship graph
        elif p == "/intel/graph":
            self._json(pattern_engine.get_entity_graph())

        # /intel/pol_briefing → Palantir-style pattern-of-life briefing
        elif p == "/intel/pol_briefing":
            self._json({"briefing": pattern_engine.get_pol_briefing()})

        # /intel/wanted       → FBI wanted persons database
        elif p == "/intel/wanted":
            q = qp("q")
            if q:
                self._json({"results": face_intel.search_wanted(q), "query": q})
            else:
                self._json({
                    "wanted": face_intel.get_wanted_list(50),
                    "stats":  face_intel.get_stats(),
                })

        # /intel/match_log    → face comparison match history
        elif p == "/intel/match_log":
            self._json({"matches": face_intel.get_match_log(50)})

        # /intel/movement/<id> → cross-camera movement chain
        elif parts[0] == "intel" and len(parts) == 3 and parts[1] == "movement":
            try:
                pid   = int(parts[2])
                hours = float(qp("hours") or 24)
                self._json({"chain": pattern_engine.get_movement_chain(pid, hours),
                            "profile_id": pid})
            except ValueError:
                self._not_found()

        # /trafficcam         → neighborhood overwatch map (JPEG)
        elif p == "/trafficcam":
            img = traffic_cam.get_image()
            if not img:
                self._not_found(); return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(img)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", f"max-age={traffic_cam.CACHE_SEC}")
            self.end_headers()
            self.wfile.write(img)

        # /trafficcam/status  → traffic cam module status JSON
        elif p == "/trafficcam/status":
            self._json(traffic_cam.get_status())

        # ─── Camera framework: adapters + discovery ────────────────────
        elif p == "/api/adapters":
            self._json({
                "adapters":  camera_adapters.list_adapters(),
                "summary":   camera_adapters.adapter_summary(),
            })

        elif p == "/api/discover":
            qs     = parse_qs(parsed.query)
            subnet = qs.get("subnet", [None])[0]
            try:
                found = camera_discover.discover_all(subnet=subnet, timeout=3.0)
                self._json({
                    "count":   len(found),
                    "cameras": [c.__dict__ for c in found],
                    "yaml":    camera_discover.to_yaml_block(found),
                })
            except Exception as e:
                self._json({"error": str(e), "count": 0, "cameras": []})

        elif p == "/api/discover/yaml":
            try:
                found = camera_discover.discover_all()
                body  = camera_discover.to_yaml_block(found).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/yaml")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
            except Exception:
                self._not_found()

        # ─── Entity resolution endpoints ───────────────────────────────
        elif p == "/intel/entities":
            r = entity_resolution.get_resolver()
            self._json({
                "stats":    r.stats(),
                "entities": r.all_entities(limit=100),
                "briefing": entity_resolution.briefing(),
            })

        elif parsed.path.startswith("/intel/entity/"):
            eid = parsed.path.split("/intel/entity/", 1)[1].rstrip("/")
            if not eid or len(eid) > 64 or not all(c.isalnum() or c in "-_" for c in eid):
                self._not_found(); return
            detail = entity_resolution.get_resolver().entity_detail(eid)
            if not detail:
                self._not_found(); return
            self._json(detail)

        elif p == "/intel/merge_log":
            self._json({
                "log": entity_resolution.get_resolver().merge_log(limit=50),
            })

        # ─── Forward intelligence endpoints ─────────────────────────────
        # ─── Notification engine ────────────────────────────────────────
        elif p == "/api/notify/status":
            self._json(notifier.get_status())

        elif p == "/api/notify/test":
            self._json(notifier.test_notify())

        # ─── Evidence export ────────────────────────────────────────────
        elif parsed.path.startswith("/api/evidence/profile/"):
            raw = parsed.path.split("/api/evidence/profile/", 1)[1].rstrip("/")
            try:
                pid   = int(raw)
                hours = float(parse_qs(parsed.query).get("hours", ["72"])[0])
                zdata = evidence_export.generate_package(profile_id=pid, hours=hours)
                fname = evidence_export.package_filename(f"profile-{pid}")
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Length", str(len(zdata)))
                self.send_header("Content-Disposition", f'attachment; filename="{fname}"')
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(zdata)
            except ValueError:
                self._not_found()
            except Exception as e:
                self._json({"error": str(e)})

        elif parsed.path.startswith("/api/evidence/"):
            eid = parsed.path.split("/api/evidence/", 1)[1].rstrip("/")
            if not eid or len(eid) > 64 or not all(c.isalnum() or c in "-_" for c in eid):
                self._not_found(); return
            try:
                hours = float(parse_qs(parsed.query).get("hours", ["72"])[0])
                zdata = evidence_export.generate_package(entity_id=eid, hours=hours)
                fname = evidence_export.package_filename(eid)
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Length", str(len(zdata)))
                self.send_header("Content-Disposition", f'attachment; filename="{fname}"')
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(zdata)
            except Exception as e:
                self._json({"error": str(e)})

        elif p == "/intel/forecast":
            self._json({
                "scenarios": forward_intel.build_scenarios(),
                "briefing":  forward_intel.forecast_briefing(),
            })

        elif p == "/intel/behavior":
            self._json({"briefing": forward_intel.behavior_briefing()})

        # ─── Drone operations planner ─────────────────────────────────
        elif p == "/drone/status":
            self._json(drone_ops.status())

        elif parsed.path.startswith("/intel/classify/"):
            eid = parsed.path.split("/intel/classify/", 1)[1].rstrip("/")
            if not eid or len(eid) > 64 or not all(c.isalnum() or c in "-_" for c in eid):
                self._not_found(); return
            self._json(forward_intel.classify_entity(eid))

        # ─── SD card backfill ───────────────────────────────────────────
        elif parsed.path.startswith("/api/backfill/status/") or parsed.path.startswith("/backfill/status/"):
            prefix = "/api/backfill/status/" if parsed.path.startswith("/api/backfill/status/") else "/backfill/status/"
            job_id = parsed.path.split(prefix, 1)[1].rstrip("/")
            status = backfill_tapo.get_job_status(job_id)
            if status is None:
                self._not_found(); return
            safe = {k: v for k, v in status.items() if k != "log"}
            self._json(safe)

        elif p in ("/api/backfill/jobs", "/backfill/jobs"):
            self._json({"jobs": [
                {k: v for k, v in j.items() if k != "log"}
                for j in backfill_tapo.list_jobs()
            ]})

        # ─── Scene memory / attention zones ────────────────────────────
        elif p in ("/scene/zones", "/api/scene/zones"):
            cam = qp("camera")
            if cam:
                self._json({"camera": cam, "attention_zones": _attention_zones.get(cam, [])})
            else:
                self._json({"attention_zones": _attention_zones})

        # ─── Vision tools / scene intelligence ────────────────────────
        elif p in ("/vision/capabilities", "/api/vision/capabilities"):
            self._json(vision_tools.capabilities())

        elif parsed.path.startswith("/vision/scene/") or parsed.path.startswith("/api/vision/scene/"):
            prefix = "/api/vision/scene/" if parsed.path.startswith("/api/vision/scene/") else "/vision/scene/"
            cam_id = parsed.path.split(prefix, 1)[1].rstrip("/")
            if cam_id not in cameras:
                self._not_found(); return
            self._json(cameras[cam_id].last_vision or vision_tools.scene_metrics(
                cam_id,
                cameras[cam_id].last_detections,
                _attention_zones.get(cam_id, []),
            ))

        # ─── Field Scan (phone app) ─────────────────────────────────────
        elif p == "/scan/history":
            limit = min(int(parse_qs(parsed.query).get("limit", ["50"])[0]), 200)
            self._json(event_db.get_manual_scans(limit))

        # ─── Storage management ─────────────────────────────────────────

        # GET /media/stats
        elif p == "/media/stats":
            stats = event_db.get_media_stats()
            # add archive disk usage
            arc_bytes = 0
            arc_files = 0
            for f in _archive_files():
                arc_bytes += f.stat().st_size
                arc_files += 1
            orphans = _orphaned_archive_media(dry_run=True)
            stats["archive_bytes"] = arc_bytes
            stats["archive_files"] = arc_files
            stats["orphan_files"] = orphans["orphan_files"]
            stats["orphan_bytes"] = orphans["orphan_bytes"]
            stats["orphan_sample"] = orphans["sample"]
            stats["retention_days"] = ARCHIVE_RETENTION_DAYS
            self._json(stats)

        # GET /media/archive?camera=&limit=
        elif p == "/media/archive":
            cam    = qp("camera")
            limit  = min(int(qp("limit") or 50), 500)
            events = event_db.get_recent_events(cam, limit)
            events = [e for e in events if e.get("archived_clip") or e.get("archived_snap")]
            self._json({"events": events, "count": len(events)})

        # GET /archived/snap/<event_id>  → serve archived snapshot
        elif parts[0] == "archived" and len(parts) == 3 and parts[1] == "snap":
            try:
                eid = int(parts[2])
            except ValueError:
                self._not_found(); return
            rows = event_db.get_recent_events(limit=1)  # just to reuse conn pattern
            with __import__("sqlite3").connect(str(event_db.DB_PATH)) as _c:
                _c.row_factory = __import__("sqlite3").Row
                row = _c.execute("SELECT archived_snap FROM events WHERE id = ?", (eid,)).fetchone()
            if not row or not row["archived_snap"]:
                self._not_found(); return
            p_file = Path(row["archived_snap"])
            if not p_file.exists():
                self._not_found(); return
            data = p_file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        # GET /archived/clip/<event_id>  → serve archived clip
        elif parts[0] == "archived" and len(parts) == 3 and parts[1] == "clip":
            try:
                eid = int(parts[2])
            except ValueError:
                self._not_found(); return
            with __import__("sqlite3").connect(str(event_db.DB_PATH)) as _c:
                _c.row_factory = __import__("sqlite3").Row
                row = _c.execute("SELECT archived_clip FROM events WHERE id = ?", (eid,)).fetchone()
            if not row or not row["archived_clip"]:
                self._not_found(); return
            p_file = Path(row["archived_clip"])
            if not p_file.exists():
                self._not_found(); return
            data = p_file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        else:
            self._not_found()

    def do_PATCH(self) -> None:
        if not self._auth_ok():
            self._unauthorized(); return
        parsed = urlparse(self.path)
        parts  = parsed.path.strip("/").split("/")

        # PATCH /profile/<id>/label  {"label": "..."}
        if parts[0] == "profile" and len(parts) == 3 and parts[2] == "label":
            try:
                pid = int(parts[1])
            except ValueError:
                self._not_found(); return
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                data  = json.loads(body)
                label = str(data.get("label", "")).strip()[:48]
            except Exception:
                self.send_response(400); self.end_headers(); return
            ok = profiler.set_profile_label(pid, label)
            self._json({"ok": ok, "profile_id": pid, "label": label})
        else:
            self._not_found()

    def do_POST(self) -> None:
        if not self._auth_ok():
            self._unauthorized(); return
        parsed = urlparse(self.path)
        p      = parsed.path.rstrip("/")
        parts  = parsed.path.strip("/").split("/")

        # POST /profile/<id>/(trust|ignore|delete|merge|enroll-face)
        if parts[0] == "profile" and len(parts) == 3 and parts[2] in (
            "trust", "ignore", "delete", "merge", "enroll-face"
        ):
            try:
                pid = int(parts[1])
            except ValueError:
                self._not_found(); return
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                data = json.loads(body) if body else {}
            except Exception:
                data = {}

            if parts[2] == "trust":
                name = str(data.get("label") or profiler.get_profile_label(pid)).strip()[:36]
                if not name:
                    name = f"PROFILE-{pid:03d}"
                label = name if name.startswith("TRUSTED") else f"TRUSTED-{name}"
                ok = profiler.set_profile_label(pid, label)
                self._json({"ok": ok, "profile_id": pid, "label": label}); return

            if parts[2] == "ignore":
                reason = str(data.get("reason") or data.get("label") or "BACKGROUND").strip()[:36]
                label = reason if reason.startswith(("IGNORE", "BACKGROUND")) else f"IGNORE-{reason}"
                ok = profiler.set_profile_label(pid, label)
                self._json({"ok": ok, "profile_id": pid, "label": label}); return

            if parts[2] == "delete":
                ok = event_db.delete_profile(pid)
                self._json({"ok": ok, "profile_id": pid}); return

            if parts[2] == "merge":
                try:
                    keep_id = int(data.get("into"))
                except Exception:
                    self._json({"ok": False, "error": "missing merge target"}); return
                if keep_id == pid:
                    self._json({"ok": False, "error": "cannot merge profile into itself"}); return
                keep = event_db.get_profile(keep_id)
                drop = event_db.get_profile(pid)
                if not keep or not drop:
                    self._json({"ok": False, "error": "profile not found"}); return
                try:
                    keep_emb = json.loads(keep["embedding"])
                    drop_emb = json.loads(drop["embedding"])
                    keep_n = max(int(keep.get("sightings") or 1), 1)
                    drop_n = max(int(drop.get("sightings") or 1), 1)
                    if len(keep_emb) == len(drop_emb):
                        merged_emb = [
                            ((a * keep_n) + (b * drop_n)) / (keep_n + drop_n)
                            for a, b in zip(keep_emb, drop_emb)
                        ]
                    else:
                        merged_emb = keep_emb
                except Exception:
                    merged_emb = json.loads(keep["embedding"])
                cams = sorted(set(json.loads(keep["cameras"]) + json.loads(drop["cameras"])))
                event_db.merge_profiles(keep_id, pid, cams, merged_emb)
                self._json({"ok": True, "kept": keep_id, "merged": pid}); return

            if parts[2] == "enroll-face":
                thumb = event_db.get_profile_thumb(pid)
                if not thumb:
                    self._json({"ok": False, "error": "profile has no thumbnail"}); return
                label = str(data.get("label") or profiler.get_profile_label(pid)).strip()[:48] or f"PROFILE-{pid:03d}"
                notes = str(data.get("notes") or f"Enrolled from profile {pid}").strip()[:200]
                out_dir = Path(os.environ.get("FACE_CACHE_DIR", "/tmp/face_intel")) / "poi"
                out_dir.mkdir(parents=True, exist_ok=True)
                photo_path = out_dir / f"profile_{pid}.jpg"
                photo_path.write_bytes(thumb)
                poi = face_intel.get_engine().add_poi(
                    label=label, photo_path=str(photo_path), notes=notes, threat_level="TRUSTED"
                )
                self._json({"ok": True, "profile_id": pid, "poi": poi}); return

        # POST /scan/plate  — Field Scan LPR
        if p == "/scan/plate":
            image_bytes = _parse_upload(self)
            if not image_bytes:
                self.send_response(400); self.end_headers(); return
            result    = _field_scan_plate(image_bytes)
            plate     = result.get("plate", "")
            conf      = result.get("confidence", 0.0)
            hit       = lpr_engine.is_watched(plate) if plate else False
            label     = lpr_engine.get_plate_label(plate) if plate else None
            history   = []
            try:
                history = pattern_engine.get_history(plate) if plate else []
            except Exception:
                pass
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if plate:
                event_db.log_manual_scan("plate", plate=plate, confidence=conf,
                                         watchlist_hit=hit, timestamp=ts)
            self._json({
                "plate":           plate,
                "confidence":      conf,
                "watchlist_hit":   hit,
                "watchlist_label": label,
                "pattern_history": history,
                "timestamp":       ts,
            })

        # POST /scan/face  — Field Scan FBI cross-reference
        elif p == "/scan/face":
            image_bytes = _parse_upload(self)
            if not image_bytes:
                self.send_response(400); self.end_headers(); return
            result = _field_scan_face(image_bytes)
            ts     = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            event_db.log_manual_scan("face",
                                     fbi_match=result.get("fbi_match", False),
                                     match_name=result.get("match_name"),
                                     timestamp=ts)
            result["timestamp"] = ts
            self._json(result)

        # POST /gait/label/<id>  {"label": "..."}
        elif parts[0] in ("gait", "api") and "gait" in parts and "label" in parts:
            try:
                gait_id = int(parts[parts.index("label") - 1])
            except (ValueError, IndexError):
                self._not_found(); return
            length = int(self.headers.get("Content-Length", 0))
            try:
                data  = json.loads(self.rfile.read(length))
                label = str(data.get("label", "")).strip()[:48].upper()
            except Exception:
                self.send_response(400); self.end_headers(); return
            if not label:
                self._json({"ok": False, "error": "no label"}); return
            ok = gait_engine.label_gait_profile(gait_id, label)
            self._json({"ok": ok, "gait_id": gait_id, "label": label})

        # POST /watchlist/add  {"plate": "...", "label": "..."}
        elif p == "/watchlist/add":
            length = int(self.headers.get("Content-Length", 0))
            try:
                data  = json.loads(self.rfile.read(length))
                plate = lpr_engine._normalize_plate(str(data.get("plate", "")))
                label = str(data.get("label", "FLAGGED"))[:48]
            except Exception:
                self.send_response(400); self.end_headers(); return
            if not plate:
                self._json({"ok": False, "error": "no plate"}); return
            lpr_engine.add_watched_plate(plate, label)
            self._json({"ok": True, "plate": plate, "label": label})

        # POST /watchlist/remove  {"plate": "..."}
        elif p == "/watchlist/remove":
            length = int(self.headers.get("Content-Length", 0))
            try:
                data  = json.loads(self.rfile.read(length))
                plate = lpr_engine._normalize_plate(str(data.get("plate", "")))
            except Exception:
                self.send_response(400); self.end_headers(); return
            if not plate:
                self._json({"ok": False, "error": "no plate"}); return
            lpr_engine.remove_watched_plate(plate)
            self._json({"ok": True, "plate": plate})

        # POST /agent/query  {"text": "...", "camera": "..."}
        elif p == "/agent/query":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                data   = json.loads(body)
                text   = str(data.get("text", "")).strip()[:512]
                camera = data.get("camera") or None
            except Exception:
                self.send_response(400); self.end_headers(); return
            if not text:
                self._json({"error": "empty query"}); return
            result = query_agent.query(text, camera)
            self._json(result)

        # POST /media/purge  {"days": 14, "dry_run": false}
        elif p == "/media/purge":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                data    = json.loads(body) if body else {}
                days    = int(data.get("days", ARCHIVE_RETENTION_DAYS))
                dry_run = bool(data.get("dry_run", False))
            except Exception:
                days = ARCHIVE_RETENTION_DAYS; dry_run = False
            result = _purge_old_media(days=days, dry_run=dry_run)
            self._json(result)

        # POST /media/orphans  {"dry_run": false}
        elif p == "/media/orphans":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                data = json.loads(body) if body else {}
                dry_run = bool(data.get("dry_run", True))
            except Exception:
                dry_run = True
            result = _orphaned_archive_media(dry_run=dry_run)
            self._json({"ok": True, **result})

        # POST /drone/mission  {"kind": "perimeter|recon|incident", "notes": "..."}
        elif p == "/drone/mission":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                data = json.loads(body) if body else {}
            except Exception:
                data = {}
            kind = str(data.get("kind") or "perimeter").strip().lower()
            operator = str(data.get("operator") or "dashboard").strip()
            notes = str(data.get("notes") or "").strip()
            self._json(drone_ops.start_mission(kind, operator=operator, notes=notes))

        # POST /drone/abort  {"reason": "..."}
        elif p == "/drone/abort":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                data = json.loads(body) if body else {}
            except Exception:
                data = {}
            self._json(drone_ops.abort_mission(str(data.get("reason") or "operator abort")))


        # POST /api/backfill/<cam_id|all>  {"hours": 24}
        elif ((parsed.path.startswith("/api/backfill/") and not parsed.path.startswith("/api/backfill/status"))
              or (parsed.path.startswith("/backfill/") and not parsed.path.startswith("/backfill/status"))):
            prefix = "/api/backfill/" if parsed.path.startswith("/api/backfill/") else "/backfill/"
            cam_id = parsed.path.split(prefix, 1)[1].rstrip("/")
            # Look up the camera config
            cfg_list = load_config()
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                data = json.loads(body) if body else {}
                hours = float(data.get("hours", 24))
                days = float(data.get("days", hours / 24))
            except Exception:
                hours = 24.0
                days = 1.0
            hours = max(1.0, min(hours, 24 * 90))

            if cam_id in ("all", "*"):
                tapo_cfgs = [c for c in cfg_list if c.get("type") == "tapo" and c.get("id") in cameras]
                if not tapo_cfgs:
                    self._json({"error": "no Tapo cameras configured for SD backfill", "jobs": []}); return
                jobs = [
                    {"cam": cfg["id"], "job_id": backfill_tapo.start_backfill_job(cfg, days=days, hours=hours)}
                    for cfg in tapo_cfgs
                ]
                self._json({"ok": True, "jobs": jobs, "hours": hours, "eligible_cameras": [j["cam"] for j in jobs]})
                return

            if cam_id not in cameras:
                self._json({"error": f"unknown camera: {cam_id}"}); return
            cam_cfg  = next((c for c in cfg_list if c["id"] == cam_id), None)
            if not cam_cfg or cam_cfg.get("type") != "tapo":
                self._json({"error": f"{cam_id} is not a Tapo camera"}); return
            job_id = backfill_tapo.start_backfill_job(cam_cfg, days=days, hours=hours)
            self._json({"ok": True, "job_id": job_id, "cam": cam_id, "hours": hours})

        # POST /events/delete  {"ids": [1,2,3]}  or {"all": true}
        elif p == "/events/delete":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                data = json.loads(body) if body else {}
            except Exception:
                data = {}
            delete_media = bool(data.get("delete_media", True))
            if data.get("all"):
                rows = event_db.get_all_event_media_rows()
                media = _delete_event_media(rows, include_latest=True) if delete_media else {"files_deleted": 0, "bytes_freed": 0}
                with __import__("sqlite3").connect(str(event_db.DB_PATH)) as _c:
                    _c.execute("DELETE FROM events")
                    deleted = _c.execute("SELECT changes()").fetchone()[0]
                self._json({"ok": True, "deleted": deleted, **media})
            else:
                ids = [int(i) for i in data.get("ids", []) if str(i).isdigit()]
                rows = event_db.get_events_by_ids(ids)
                media = _delete_event_media(rows, include_latest=False) if delete_media else {"files_deleted": 0, "bytes_freed": 0}
                deleted = event_db.purge_events(ids) if ids else 0
                self._json({"ok": True, "deleted": deleted, **media})

        # POST /event/<id>/star  {"starred": true}
        elif parsed.path.startswith("/event/") and parsed.path.endswith("/star"):
            parts = parsed.path.strip("/").split("/")
            try:
                eid = int(parts[1])
            except (ValueError, IndexError):
                self._not_found(); return
            length  = int(self.headers.get("Content-Length", 0))
            body    = self.rfile.read(length)
            try:
                data    = json.loads(body) if body else {}
                starred = bool(data.get("starred", True))
            except Exception:
                starred = True
            ok = event_db.star_event(eid, starred)
            self._json({"ok": ok, "event_id": eid, "starred": starred})

        # POST /scene/zones/<cam_id>  {"zones": [...]}
        elif ((parsed.path.startswith("/scene/zones/") or parsed.path.startswith("/api/scene/zones/"))):
            prefix = "/api/scene/zones/" if parsed.path.startswith("/api/scene/zones/") else "/scene/zones/"
            cam_id = parsed.path.split(prefix, 1)[1].rstrip("/")
            if not cam_id:
                self._not_found(); return
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                data = json.loads(body) if body else {}
                zones = data.get("zones") or data.get("attention_zones") or []
                if not isinstance(zones, list):
                    raise ValueError("zones must be a list")
            except Exception as e:
                self._json({"ok": False, "error": str(e)}); return
            _attention_zones[cam_id] = [_clean_zone(z) for z in zones]
            _save_scene_zones()
            self._json({"ok": True, "camera": cam_id, "attention_zones": _attention_zones[cam_id]})

        # POST /camera/<cam_id>/watch  {"minutes": 10}
        elif parsed.path.startswith("/camera/") and parsed.path.endswith("/watch"):
            try:
                cam_id = parsed.path.strip("/").split("/")[1]
            except Exception:
                self._not_found(); return
            if cam_id not in cameras:
                self._not_found(); return
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                data = json.loads(body) if body else {}
            except Exception:
                data = {}
            status = _start_watch_mode(
                cam_id,
                minutes=float(data.get("minutes", 10)),
                poll_interval=float(data.get("poll_interval", 3)),
                cooldown=float(data.get("cooldown", 60)),
                capture_duration=int(data.get("capture_duration", 8)),
            )
            self._json({"ok": True, "camera": cam_id, "watch": status})

        else:
            self._not_found()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Palm-Token")
        self.end_headers()

    def _cam_status(self, cam_id: str) -> dict:
        s        = cameras[cam_id]
        clip     = media_path(cam_id, "clip")
        snap     = media_path(cam_id, "snap")
        snap_ann = snap.parent / "snap_ann.jpg"
        age      = int(time.time() - s.last_seen) if s.last_seen else None
        return {
            "online":        s.online,
            "last_seen":     datetime.fromtimestamp(s.last_seen).isoformat() if s.last_seen else None,
            "last_seen_ts":  s.last_seen,
            "age_s":         age,
            "events":        s.events,
            "last_mode":     s.last_mode,
            "has_clip":      clip.exists() and clip.stat().st_size > 10_000,
            "has_snap":      snap.exists() and snap.stat().st_size > 0,
            "has_snap_ann":  snap_ann.exists() and snap_ann.stat().st_size > 0,
            "detections":    s.last_detections,
            "profiles":      s.last_profiles,
            "summary":       s.last_summary,
            "vision":        s.last_vision,
            "watch":         _watch_status(cam_id),
            "attention_zones": _attention_zones.get(cam_id, []),
            "focus_hits": [
                {
                    "class": d.get("class"),
                    "confidence": d.get("confidence"),
                    "zone": d.get("attention_zone"),
                    "priority": d.get("attention_priority"),
                }
                for d in s.last_detections if d.get("attention_zone")
            ],
        }

    def _json(self, data: object) -> None:
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Path, mime: str) -> None:
        if not path.exists() or path.stat().st_size == 0:
            self._not_found(); return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _not_found(self) -> None:
        self.send_response(404)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()


# ── Config loader ─────────────────────────────────────────────────

def load_config() -> list[dict]:
    if not CONFIG_FILE.exists():
        print(f"[config] {CONFIG_FILE} not found — falling back to env vars (single Tapo)", flush=True)
        return [{
            "id":               "doorbell",
            "name":             "Front Door D210",
            "type":             "tapo",
            "ip":               os.environ.get("TAPO_IP", "192.168.x.x"),
            "password":         os.environ.get("TAPO_PASSWORD", ""),
            "port":             8800,
            "poll_interval":    3,
            "capture_duration": 9,
        }]

    raw      = CONFIG_FILE.read_text()
    expanded = os.path.expandvars(raw)
    return yaml.safe_load(expanded)["cameras"]


# ── Entry point ───────────────────────────────────────────────────

async def main() -> None:
    event_db.init_db()
    _load_scene_zones()
    cam_cfgs = load_config()
    print(f"[watcher] Starting with {len(cam_cfgs)} camera(s)", flush=True)

    for cfg in cam_cfgs:
        zones = cfg.get("exclusion_zones", [])
        if zones:
            _exclusion_zones[cfg["id"]] = zones
            print(f"[exclusion] {cfg['id']}: {len(zones)} zone(s) loaded", flush=True)
        known = cfg.get("known_zones", [])
        if known:
            _known_zones[cfg["id"]] = known
            print(f"[known] {cfg['id']}: {len(known)} zone(s) loaded — alerts suppressed, profiled as NEIGHBOR", flush=True)
        attention = cfg.get("attention_zones", cfg.get("focus_zones", []))
        if attention:
            _attention_zones[cfg["id"]] = [_clean_zone(z) for z in attention]
            print(f"[scene] {cfg['id']}: {len(attention)} attention zone(s) loaded", flush=True)
    _save_scene_zones()

    # Start intelligence feeds background refresh
    intel_feeds.start_background_refresh(interval_s=180)

    # ── Auto-purger: clean non-starred archived media older than retention window
    async def _auto_purge_loop():
        import asyncio as _aio
        while True:
            await _aio.sleep(86400)  # run once per day
            result = _purge_old_media(days=ARCHIVE_RETENTION_DAYS)
            if result["purged_events"] > 0:
                mb = result["bytes_freed"] / 1_048_576
                print(f"[purge] auto-purged {result['purged_events']} events, freed {mb:.1f} MB", flush=True)
    asyncio.create_task(_auto_purge_loop())
    print("[watcher] Intel feeds refresh started", flush=True)

    # Start FBI wanted persons database (background fetch)
    face_intel.start_background_refresh()
    print("[watcher] Face intel (FBI database) refresh started", flush=True)

    # Start neighborhood traffic cam (OSM tile map)
    traffic_cam.start_background_refresh()
    print("[watcher] Neighborhood overwatch started", flush=True)

    # Pre-warm EasyOCR so first field scan doesn't time out
    def _prewarm_ocr():
        try:
            lpr_engine._init_ocr()
            print("[watcher] EasyOCR pre-warmed", flush=True)
        except Exception as e:
            print(f"[watcher] EasyOCR pre-warm skipped: {e}", flush=True)
    threading.Thread(target=_prewarm_ocr, daemon=True).start()

    # Initialize pattern-of-life engine
    try:
        pattern_engine.get_engine().build()
    except Exception:
        pass
    print("[watcher] Pattern-of-life engine initialized", flush=True)

    # Initialize entity resolution
    try:
        entity_resolution.get_resolver()
    except Exception as e:
        print(f"[watcher] entity_resolution init failed: {e}", flush=True)
    print("[watcher] Entity resolution engine initialized", flush=True)

    # Log adapter registry
    print(f"[watcher] {len(camera_adapters.list_adapters())} camera adapters registered", flush=True)

    for cfg in cam_cfgs:
        cameras[cfg["id"]] = CameraState(id=cfg["id"], name=cfg["name"])

    threading.Thread(
        target=lambda: HTTPServer(("", SERVE_PORT), Handler).serve_forever(),
        daemon=True,
    ).start()
    print(f"[watcher] HTTP API → http://0.0.0.0:{SERVE_PORT}", flush=True)

    tasks = []
    for cfg in cam_cfgs:
        ctype = cfg.get("type", "tapo")
        # Built-in fast-path watchers (high-frequency motion detection)
        if ctype == "tapo":
            tasks.append(asyncio.create_task(tapo_poll_loop(cfg)))
        elif ctype == "rtsp":
            tasks.append(asyncio.create_task(rtsp_poll_loop(cfg)))
        elif ctype == "go2rtc":
            tasks.append(asyncio.create_task(go2rtc_poll_loop(cfg)))
        # Universal adapter path: any registered vendor (hikvision, reolink, amcrest,
        # onvif, mjpeg, http_snap, wyze, usb, bluetooth, ring, etc.)
        elif camera_adapters.get_adapter(ctype) is not None:
            tasks.append(asyncio.create_task(adapter_poll_loop(cfg)))
        else:
            print(f"[{cfg['id']}] Unknown camera type '{ctype}' — skipped"
                  f"  (registered: {sorted([a['vendor'] for a in camera_adapters.list_adapters()])})",
                  flush=True)

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
