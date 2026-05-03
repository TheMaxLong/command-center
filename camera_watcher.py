#!/usr/bin/env python3.12
"""
PALM COMMAND — Multi-camera vision watcher v2.

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

# ── Config ────────────────────────────────────────────────────────
SERVE_PORT  = int(os.environ.get("WATCHER_PORT", "8181"))
CONFIG_FILE = Path(os.environ.get("CAMERAS_CONFIG", "/config/cameras.yaml"))
MEDIA_DIR   = Path(os.environ.get("MEDIA_DIR", "/tmp/cams"))

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

cameras: dict[str, CameraState] = {}


# ── Media helpers ─────────────────────────────────────────────────

def media_path(cam_id: str, kind: str) -> Path:
    d = MEDIA_DIR / cam_id
    d.mkdir(parents=True, exist_ok=True)
    ext = "mp4" if kind == "clip" else "jpg"
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
) -> None:
    """Run AI on snapshot, persist event + detections. Called in a daemon thread."""
    detections: list[dict] = []
    if snap_path and Path(snap_path).exists():
        detections = ai_engine.detect(snap_path)

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

    if snap_path and detections:
        ai_engine.annotate(snap_path, detections)

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

    # ── Gait analysis — skeletal biometric identification
    if snap_path and detections:
        person_dets = [d for d in detections if d.get("class") == "person"]
        if person_dets:
            try:
                detections = gait_engine.process_frame(snap_path, detections, cam_id, event_ts)
            except Exception as ge:
                print(f"[gait] error: {ge}", flush=True)

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

    # Check for stranger alerts
    for pid in profile_ids:
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

    if detections:
        ts_str = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts_str}] [{cam_id}] {summary}", flush=True)


def _fire_ai(cam_id: str, event_ts: float, clip_ok: bool, snap_ok: bool) -> None:
    clip_p = str(media_path(cam_id, "clip")) if clip_ok else None
    snap_p = str(media_path(cam_id, "snap")) if snap_ok else None
    threading.Thread(
        target=_run_ai_and_store,
        args=(cam_id, event_ts, clip_p, snap_p),
        daemon=True,
    ).start()


# ── Tapo camera watcher ───────────────────────────────────────────

async def _tapo_capture(cam_cfg: dict, cam_id: str) -> tuple[bool, bool]:
    from pytapo import HttpMediaSession
    from pytapo.const import EncryptionMethod

    ip  = cam_cfg["ip"]
    port = int(cam_cfg.get("port", 8800))
    pwd  = cam_cfg["password"]
    dur  = int(cam_cfg.get("capture_duration", 9))

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
    poll_s     = float(cam_cfg.get("poll_interval", 8))
    cooldown_s = float(cam_cfg.get("cooldown", 30))
    was_online = False
    capturing  = False
    last_cap   = 0.0
    state      = cameras[cam_id]

    print(f"[{cam_id}] Tapo watcher — {ip}:{port} every {poll_s}s cooldown={cooldown_s}s", flush=True)

    while True:
        try:
            _, w = await asyncio.wait_for(asyncio.open_connection(ip, port), timeout=3)
            w.close()
            try:
                await w.wait_closed()
            except Exception:
                pass

            in_cooldown = (time.time() - last_cap) < cooldown_s
            if not was_online and not capturing and not in_cooldown:
                was_online    = True
                capturing     = True
                state.online  = True
                state.last_seen = time.time()
                state.events += 1
                ts_str = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts_str}] [{cam_id}] MOTION #{state.events}", flush=True)

                async def _capture(cfg=cam_cfg, cid=cam_id):
                    nonlocal capturing, was_online, last_cap
                    try:
                        event_ts = time.time()
                        clip_ok, snap_ok = await _tapo_capture(cfg, cid)
                        if clip_ok or snap_ok:
                            state.last_mode = "clip" if clip_ok else "snap"
                            _fire_ai(cid, event_ts, clip_ok, snap_ok)
                        last_cap = time.time()
                    finally:
                        capturing    = False
                        was_online   = False
                        state.online = False

                asyncio.create_task(_capture())

        except Exception:
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
            if detections:
                ai_engine.annotate(snap, detections)
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
        self.send_header("WWW-Authenticate", 'Bearer realm="PALM COMMAND"')
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

        # /trends[?camera=&weeks=]  — now uses full trend_analyzer
        elif p == "/trends":
            camera = qp("camera")
            weeks  = int(qp("weeks") or 5)
            self._json(trend_analyzer.analyze(camera, weeks))

        # /intel/briefing[?camera=]
        elif p == "/intel/briefing":
            camera = qp("camera")
            self._json(intel_engine.daily_briefing(camera))

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

        elif parsed.path.startswith("/intel/classify/"):
            eid = parsed.path.split("/intel/classify/", 1)[1].rstrip("/")
            if not eid or len(eid) > 64 or not all(c.isalnum() or c in "-_" for c in eid):
                self._not_found(); return
            self._json(forward_intel.classify_entity(eid))

        # ─── Field Scan (phone app) ─────────────────────────────────────
        elif p == "/scan/history":
            limit = min(int(parse_qs(parsed.query).get("limit", ["50"])[0]), 200)
            self._json(event_db.get_manual_scans(limit))

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
    cam_cfgs = load_config()
    print(f"[watcher] Starting with {len(cam_cfgs)} camera(s)", flush=True)

    # Start intelligence feeds background refresh
    intel_feeds.start_background_refresh(interval_s=180)
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
