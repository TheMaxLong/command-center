#!/usr/bin/env python3.12
"""
PALM COMMAND — Multi-camera vision watcher.

Supersedes doorbell_watcher.py.
Supports Tapo cameras (D210/D230/C310) and generic RTSP sources (ESP32-CAM, Reolink, etc.).

HTTP API on :8181
  GET /status              → JSON dict of all cameras
  GET /status/<cam_id>     → JSON for one camera (backwards-compatible with doorbell_watcher format)
  GET /clip/<cam_id>       → latest clip  (video/mp4)
  GET /snap/<cam_id>       → latest snap  (image/jpeg)
  GET /events[?camera=&limit=] → recent events from DB (JSON)
  GET /trends[?camera=&weeks=] → heatmap + detection summary (JSON)
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
    id:               str
    name:             str
    online:           bool             = False
    last_seen:        Optional[float]  = None
    events:           int              = 0
    last_mode:        Optional[str]    = None
    last_detections:  list             = field(default_factory=list)
    last_profiles:    list             = field(default_factory=list)  # profiler hits


# Global registry filled at startup
cameras: dict[str, CameraState] = {}


# ── Media helpers ─────────────────────────────────────────────────

def media_path(cam_id: str, kind: str) -> Path:
    d = MEDIA_DIR / cam_id
    d.mkdir(parents=True, exist_ok=True)
    ext = "mp4" if kind == "clip" else "jpg"
    return d / f"{kind}.{ext}"


def _ffmpeg_from_ts(ts_data: bytes, cam_id: str, capture_s: int) -> tuple[bool, bool]:
    """Convert raw MPEG-TS bytes → clip.mp4 + snap.jpg. Returns (clip_ok, snap_ok)."""
    clip = media_path(cam_id, "clip")
    snap = media_path(cam_id, "snap")
    clip_ok = snap_ok = False
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-f", "mpegts", "-i", "pipe:0",
             "-c:v", "copy", "-an", "-movflags", "+faststart",
             "-t", str(capture_s), str(clip)],
            input=ts_data, capture_output=True, timeout=30,
        )
        clip_ok = clip.exists() and clip.stat().st_size > 10_000
    except Exception:
        pass
    try:
        r = subprocess.run(
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

    event_id = event_db.insert_event(cam_id, event_ts, clip_path, snap_path)
    if detections:
        event_db.add_detections(event_id, detections)

    # Annotate snapshot with bounding boxes
    if snap_path and detections:
        ai_engine.annotate(snap_path, detections)

    # Person profiling
    profile_ids: list[int] = []
    if snap_path and detections:
        crops = ai_engine.extract_crops(snap_path, detections)
        profile_ids = profiler.match_or_create(cam_id, event_ts, crops, event_id)

    state = cameras[cam_id]
    state.last_detections = detections
    state.last_profiles   = [
        {"id": pid, "label": profiler.get_profile_label(pid)}
        for pid in profile_ids
    ]

    if detections:
        tags = ", ".join(
            f"{d['class'].upper()} {int(d['confidence'] * 100)}%"
            for d in detections
        )
        ts_str = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts_str}] [{cam_id}] AI: {tags}", flush=True)


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

    ip      = cam_cfg["ip"]
    port    = int(cam_cfg.get("port", 8800))
    pwd     = cam_cfg["password"]
    dur     = int(cam_cfg.get("capture_duration", 9))

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
    poll_s     = float(cam_cfg.get("poll_interval", 3))
    was_online = False
    capturing  = False
    state      = cameras[cam_id]

    print(f"[{cam_id}] Tapo watcher — {ip}:{port} every {poll_s}s", flush=True)

    while True:
        try:
            _, w = await asyncio.wait_for(asyncio.open_connection(ip, port), timeout=3)
            w.close()
            try:
                await w.wait_closed()
            except Exception:
                pass

            if not was_online and not capturing:
                was_online      = True
                capturing       = True
                state.online    = True
                state.last_seen = time.time()
                state.events   += 1
                ts_str = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts_str}] [{cam_id}] MOTION #{state.events}", flush=True)

                async def _capture(cfg=cam_cfg, cid=cam_id):
                    nonlocal capturing, was_online
                    try:
                        event_ts = time.time()
                        clip_ok, snap_ok = await _tapo_capture(cfg, cid)
                        if clip_ok or snap_ok:
                            state.last_mode = "clip" if clip_ok else "snap"
                            _fire_ai(cid, event_ts, clip_ok, snap_ok)
                    finally:
                        capturing  = False
                        was_online = False
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
    """
    Periodically grabs a JPEG frame from go2rtc's /api/frame.jpeg endpoint
    and runs AI detection on it.  No motion detection — just continuous
    periodic inference so the dashboard always shows what's in frame.

    Config keys:
      go2rtc_url   : base URL of go2rtc  (default http://go2rtc:1984)
      source_name  : stream name in go2rtc (default = camera id)
      ai_interval  : seconds between AI runs (default 10)
    """
    import urllib.request, urllib.error

    cam_id      = cam_cfg["id"]
    base_url    = cam_cfg.get("go2rtc_url", "http://go2rtc:1984").rstrip("/")
    src_name    = cam_cfg.get("source_name", cam_id)
    interval    = float(cam_cfg.get("ai_interval", 10))
    state       = cameras[cam_id]
    snap        = media_path(cam_id, "snap")

    print(f"[{cam_id}] go2rtc AI watcher — {base_url}/api/frame.jpeg?src={src_name} every {interval}s", flush=True)

    while True:
        await asyncio.sleep(interval)
        try:
            url = f"{base_url}/api/frame.jpeg?src={src_name}"
            with urllib.request.urlopen(url, timeout=5) as r:  # nosec — localhost only
                data = r.read()
            if len(data) < 512:
                continue
            snap.parent.mkdir(parents=True, exist_ok=True)
            snap.write_bytes(data)
            state.online    = True
            state.last_seen = time.time()

            detections = ai_engine.detect(snap)

            # Annotate snapshot with bounding boxes
            if detections:
                ai_engine.annotate(snap, detections)

            # Person profiling
            profile_ids: list[int] = []
            if detections:
                crops = ai_engine.extract_crops(snap, detections)
                profile_ids = profiler.match_or_create(cam_id, state.last_seen or event_ts, crops)

            state.last_detections = detections
            state.last_profiles   = [
                {"id": pid, "label": profiler.get_profile_label(pid)}
                for pid in profile_ids
            ]

            if detections:
                tags = ", ".join(
                    f"{d['class'].upper()} {int(d['confidence']*100)}%"
                    for d in detections
                )
                ts_str = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts_str}] [{cam_id}] AI: {tags}", flush=True)

        except urllib.error.URLError:
            state.online = False
        except Exception as e:
            print(f"[{cam_id}] go2rtc AI error: {e}", flush=True)
            state.online = False


# ── RTSP camera watcher ───────────────────────────────────────────

async def rtsp_poll_loop(cam_cfg: dict) -> None:
    """
    Motion detection via frame-differencing for generic RTSP cameras.
    Requires opencv-python-headless and numpy.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print(
            f"[{cam_cfg['id']}] RTSP watcher needs opencv-python-headless and numpy. "
            "Add them to requirements.txt.",
            flush=True,
        )
        return

    cam_id    = cam_cfg["id"]
    rtsp_url  = cam_cfg["rtsp_url"]
    poll_s    = float(cam_cfg.get("poll_interval", 5))
    threshold = float(cam_cfg.get("motion_threshold", 0.02))
    dur       = int(cam_cfg.get("capture_duration", 9))
    state     = cameras[cam_id]

    print(f"[{cam_id}] RTSP watcher — {rtsp_url}", flush=True)

    prev_gray  = None
    capturing  = False

    while True:
        try:
            cap   = cv2.VideoCapture(rtsp_url)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                state.online = False
                await asyncio.sleep(poll_s)
                continue

            state.online = True
            gray = cv2.GaussianBlur(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0
            )

            if prev_gray is not None and not capturing:
                diff  = cv2.absdiff(prev_gray, gray)
                score = float(np.count_nonzero(diff > 25)) / diff.size
                if score > threshold:
                    capturing        = True
                    state.events    += 1
                    state.last_seen  = time.time()
                    event_ts         = time.time()
                    ts_str = datetime.now().strftime("%H:%M:%S")
                    print(
                        f"[{ts_str}] [{cam_id}] MOTION #{state.events} (score={score:.3f})",
                        flush=True,
                    )

                    def _rtsp_capture(url=rtsp_url, cid=cam_id, ets=event_ts, d=dur):
                        nonlocal capturing
                        try:
                            # Prefer HTTP snapshot endpoint if provided
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


# ── HTTP server ───────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a) -> None:
        pass  # silence per-request logs

    def do_GET(self) -> None:
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
                self._not_found()
                return
            self._json(self._cam_status(cam_id))

        # /clip/<cam_id>  or  /snap/<cam_id>  or  /snap_ann/<cam_id>
        elif parts[0] in ("clip", "snap", "snap_ann") and len(parts) == 2:
            cam_id = parts[1]
            kind   = parts[0]
            if cam_id not in cameras:
                self._not_found()
                return
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
            except (ValueError, Exception):
                thumb = None
            if not thumb:
                self._not_found()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(thumb)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(thumb)

        # /events[?camera=&limit=]
        elif p == "/events":
            camera = qp("camera")
            limit  = int(qp("limit") or 50)
            self._json(event_db.get_recent_events(camera, limit))

        # /trends[?camera=&weeks=]
        elif p == "/trends":
            camera = qp("camera")
            weeks  = int(qp("weeks") or 5)
            self._json({
                "heatmap":    event_db.get_hourly_heatmap(camera, weeks),
                "detections": event_db.get_detection_summary(camera, weeks),
            })

        else:
            self._not_found()

    def _cam_status(self, cam_id: str) -> dict:
        s       = cameras[cam_id]
        clip    = media_path(cam_id, "clip")
        snap    = media_path(cam_id, "snap")
        snap_ann = snap.parent / "snap_ann.jpg"
        age     = int(time.time() - s.last_seen) if s.last_seen else None
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
            self._not_found()
            return
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
    expanded = os.path.expandvars(raw)     # expand ${TAPO_IP} etc.
    return yaml.safe_load(expanded)["cameras"]


# ── Entry point ───────────────────────────────────────────────────

async def main() -> None:
    event_db.init_db()
    cam_cfgs = load_config()
    print(f"[watcher] Starting with {len(cam_cfgs)} camera(s)", flush=True)

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
        if ctype == "tapo":
            tasks.append(asyncio.create_task(tapo_poll_loop(cfg)))
        elif ctype == "rtsp":
            tasks.append(asyncio.create_task(rtsp_poll_loop(cfg)))
        elif ctype == "go2rtc":
            tasks.append(asyncio.create_task(go2rtc_poll_loop(cfg)))
        else:
            print(f"[{cfg['id']}] Unknown camera type '{ctype}' — skipped", flush=True)

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
