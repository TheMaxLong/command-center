#!/usr/bin/env python3.12
"""
D210 doorbell watcher.
Polls port 8800 every 3s. On wake: captures one 9s clip via pytapo,
saves to /tmp/doorbell_clip.mp4 (always overwritten — no bloat).
Serves status + media on :8181 for the dashboard.
"""
import asyncio, json, os, subprocess, time, threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from pytapo import HttpMediaSession
from pytapo.const import EncryptionMethod

IP          = os.environ.get("TAPO_IP", "192.168.x.x")
PORT        = 8800
CLOUD_PASS  = os.environ.get("TAPO_PASSWORD", "")
POLL_S      = 3
CAPTURE_S   = 9
CLIP_PATH   = Path("/tmp/doorbell_clip.mp4")
SNAP_PATH   = Path("/tmp/doorbell_snap.jpg")
STATUS_PATH = Path("/tmp/doorbell_status.json")
SERVE_PORT  = 8181

PREVIEW_REQ = json.dumps({
    "type": "request", "seq": 1,
    "params": {"preview": {"audio": ["default"], "channels": [0], "resolutions": ["HD"]}, "method": "get"},
})

status = {"online": False, "last_seen": None, "events": 0, "last_mode": None}

def write_status():
    age = int(time.time() - status["last_seen"]) if status["last_seen"] else None
    STATUS_PATH.write_text(json.dumps({
        "online":       status["online"],
        "last_seen":    datetime.fromtimestamp(status["last_seen"]).isoformat() if status["last_seen"] else None,
        "last_seen_ts": status["last_seen"],
        "age_s":        age,
        "events":       status["events"],
        "last_mode":    status["last_mode"],
        "has_clip":     CLIP_PATH.exists() and CLIP_PATH.stat().st_size > 10_000,
        "has_snap":     SNAP_PATH.exists() and SNAP_PATH.stat().st_size > 0,
    }))

def save_clip(ts_data: bytes) -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "mpegts", "-i", "pipe:0",
             "-c:v", "copy", "-an", "-movflags", "+faststart",
             "-t", str(CAPTURE_S), str(CLIP_PATH)],
            input=ts_data, capture_output=True, timeout=30,
        )
        return CLIP_PATH.exists() and CLIP_PATH.stat().st_size > 10_000
    except Exception:
        return False

def save_snapshot(ts_data: bytes) -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "mpegts", "-i", "pipe:0",
             "-vframes", "1", "-q:v", "2", "-f", "image2", str(SNAP_PATH)],
            input=ts_data, capture_output=True, timeout=20,
        )
        return SNAP_PATH.exists() and SNAP_PATH.stat().st_size > 0
    except Exception:
        return False

async def capture_clip():
    session = HttpMediaSession(
        ip=IP, cloud_password=CLOUD_PASS, super_secret_key="",
        encryptionMethod=EncryptionMethod.SHA256, port=PORT, window_size=50,
    )
    ts_buf = bytearray()
    try:
        await asyncio.wait_for(session.start(), timeout=8)
        deadline = time.monotonic() + CAPTURE_S
        async for resp in session.transceive(PREVIEW_REQ, no_data_timeout=4.0):
            if resp.mimetype == "video/mp2t" and isinstance(resp.plaintext, bytes):
                ts_buf.extend(resp.plaintext)
            if time.monotonic() >= deadline:
                break
    except Exception as e:
        print(f"  stream: {e}", flush=True)
    finally:
        try: await session.close()
        except: pass

    if len(ts_buf) < 4096:
        print(f"  only {len(ts_buf)}b — no clip", flush=True)
        return

    raw = bytes(ts_buf)
    if save_clip(raw):
        save_snapshot(raw)  # poster frame
        status["last_mode"] = "clip"
    elif save_snapshot(raw):
        status["last_mode"] = "snap"
    else:
        print("  ffmpeg failed", flush=True)
        return

    status["last_seen"] = time.time()
    write_status()
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {status['last_mode']} saved  ({len(ts_buf)//1024} KB)", flush=True)

async def poll_loop():
    print(f"Doorbell watcher — polling {IP}:{PORT} every {POLL_S}s", flush=True)
    was_online = False
    capturing  = False

    while True:
        try:
            _, w = await asyncio.wait_for(asyncio.open_connection(IP, PORT), timeout=3)
            w.close()
            try: await w.wait_closed()
            except: pass

            if not was_online and not capturing:
                was_online = True
                capturing  = True
                status["online"]    = True
                status["last_seen"] = time.time()
                status["events"]   += 1
                write_status()
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] MOTION #{status['events']} — capturing...", flush=True)

                async def do_capture():
                    nonlocal capturing, was_online
                    try:
                        await capture_clip()
                    finally:
                        capturing  = False
                        was_online = False

                asyncio.create_task(do_capture())

        except Exception:
            if was_online and not capturing:
                status["online"] = False
                write_status()
            if not capturing:
                was_online = False

        await asyncio.sleep(POLL_S)

# ── HTTP server ───────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        p = self.path.split("?")[0]
        if   p == "/clip":   self._serve(CLIP_PATH, "video/mp4")
        elif p == "/snap":   self._serve(SNAP_PATH, "image/jpeg")
        elif p == "/status":
            write_status()
            self._serve(STATUS_PATH, "application/json")
        else:
            self.send_response(404); self.end_headers()

    def _serve(self, path, ctype):
        if path.exists() and path.stat().st_size > 0:
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(204); self.end_headers()

class ReuseServer(HTTPServer):
    allow_reuse_address = True

def run_http():
    srv = ReuseServer(("", SERVE_PORT), Handler)
    print(f"Media server on :{SERVE_PORT}", flush=True)
    srv.serve_forever()

if __name__ == "__main__":
    write_status()
    threading.Thread(target=run_http, daemon=True).start()
    asyncio.run(poll_loop())
