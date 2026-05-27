#!/usr/bin/env python3.12
"""
COMMAND CENTER — Dashboard server for Replit.
Serves the static dashboard on port 5000 and proxies:
  /api/*     → camera watcher on :8181
  /go2rtc/*  → go2rtc on :1984
"""
import os
import socket
import threading
import urllib.request
import urllib.error
from http.server import SimpleHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path

DASHBOARD_DIR = Path(__file__).parent / "dashboard"
SERVE_PORT = 8888
WATCHER_URL = "http://localhost:8181"
GO2RTC_URL = "http://localhost:1984"

# Sentinel digest tail synced from Pixel 6 by ~/bin/sentinel-digest-sync.sh (5min).
SENTINEL_DIGEST_TAIL = Path.home() / ".local" / "state" / "sentinel-digest.tail"
SENTINEL_DIGEST_SYNCED = Path.home() / ".local" / "state" / "sentinel-digest.tail.synced"

# Satellite image library — files written by the groundstation NOAA capture
# pipeline (see ~/bin/groundstation + ~/.groundstation/spool_event.py).
# Served read-only at /noaa/<sat>-<datetime>/<filename>. Constrained to the
# canonical directory so a crafted path can't escape via .. or symlinks.
NOAA_CAPTURE_ROOT = (Path.home() / "Documents" / "noaa-captures").resolve()


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def log_message(self, fmt, *args):
        pass  # suppress per-request logs

    def do_GET(self):
        if (self.headers.get("Upgrade", "").lower() == "websocket"
                and self.path.startswith("/go2rtc/")):
            self._proxy_websocket(self.path[7:])
            return
        if self.path.rstrip("/") == "/api/sentinel-digest" or self.path.startswith("/api/sentinel-digest?"):
            self._serve_sentinel_digest()
            return
        if self.path.rstrip("/") == "/api/local/doorbell-brightness":
            self._serve_doorbell_brightness()
            return
        if self.path.rstrip("/") == "/api/local/operator":
            self._serve_operator_position()
            return
        if self.path.rstrip("/") == "/api/local/chalk":
            self._serve_chalk()
            return
        if self.path.startswith("/api/"):
            self._proxy(WATCHER_URL, self.path[4:])  # strip /api
        elif self.path.startswith("/archived/"):
            self._proxy(WATCHER_URL, self.path)       # archived snap/clip media
        elif self.path.startswith("/noaa/"):
            self._serve_noaa(self.path[len("/noaa/"):])
        elif self.path.startswith("/go2rtc/"):
            self._proxy(GO2RTC_URL, self.path[7:])   # strip /go2rtc
        elif self.path.rstrip("/") == "/monitor" or self.path.startswith("/monitor?"):
            monitor = DASHBOARD_DIR / "monitor.html"
            self._serve_file(monitor, "text/html")
        elif self.path.rstrip("/") == "/mobile" or self.path.startswith("/mobile?"):
            self._serve_file(DASHBOARD_DIR / "mobile.html", "text/html")
        elif self.path.rstrip("/") == "/manifest.json":
            self._serve_file(DASHBOARD_DIR / "manifest.json", "application/manifest+json")
        elif self.path.rstrip("/") == "/sw.js":
            self._serve_file(DASHBOARD_DIR / "sw.js", "application/javascript")
        elif self.path.rstrip("/") == "/field" or self.path.startswith("/field?"):
            self._serve_file(DASHBOARD_DIR / "field.html", "text/html")
        elif self.path.rstrip("/") == "/field-sw.js":
            resp_path = DASHBOARD_DIR / "field-sw.js"
            if resp_path.exists():
                data = resp_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/javascript")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Service-Worker-Allowed", "/")
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404); self.end_headers()
        elif self.path.rstrip("/") == "/field-manifest.json":
            self._serve_file(DASHBOARD_DIR / "field-manifest.json", "application/manifest+json")
        else:
            super().do_GET()

    def do_POST(self):
        if self.path.rstrip("/") in ("/api/local/chalk/pin", "/api/local/chalk/unpin"):
            self._chalk_update(self.path)
            return
        if self.path.startswith("/api/"):
            self._proxy_post(WATCHER_URL, self.path[4:])
        else:
            self.send_response(404); self.end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Palm-Token")
        self.end_headers()

    def do_PATCH(self):
        if self.path.startswith("/api/"):
            self._proxy_post(WATCHER_URL, self.path[4:], method="PATCH")
        else:
            self.send_response(404); self.end_headers()

    def _serve_chalk(self):
        """Return AUTO (LOAD-BEARING) + MANUAL chalk pins as JSON."""
        import json as _json
        import re as _re

        MEMORY_PATH = (Path.home() / ".claude" / "projects" / "-Users-max"
                       / "memory" / "MEMORY.md")
        PINS_PATH = Path.home() / ".local" / "state" / "chalk-pins.json"

        auto_pins = []
        if MEMORY_PATH.exists():
            for line in MEMORY_PATH.read_text(errors="replace").splitlines():
                line = line.strip()
                if not line.startswith("- ["):
                    continue
                m = _re.match(r"^- \[(.+?)\]\((.+?)\)\s*[—\-]+\s*(.+)$", line)
                if not m:
                    continue
                title, file_, hook = m.group(1), m.group(2), m.group(3)
                if "LOAD-BEARING" in hook or "LOAD-BEARING" in title:
                    auto_pins.append({"type": "auto", "title": title,
                                      "file": file_, "hook": hook})

        manual_pins = []
        if PINS_PATH.exists():
            try:
                manual_pins = _json.loads(PINS_PATH.read_text())
            except Exception:
                manual_pins = []

        body = _json.dumps({"auto": auto_pins, "manual": manual_pins}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _chalk_update(self, path: str):
        """POST /api/local/chalk/pin or /unpin — add/remove manual pin."""
        import json as _json

        PINS_PATH = Path.home() / ".local" / "state" / "chalk-pins.json"
        PINS_PATH.parent.mkdir(parents=True, exist_ok=True)

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            entry = _json.loads(raw)
        except Exception:
            self.send_response(400); self.end_headers(); return

        pins = []
        if PINS_PATH.exists():
            try:
                pins = _json.loads(PINS_PATH.read_text())
            except Exception:
                pins = []

        if "/pin" in path and "/unpin" not in path:
            if not any(p.get("title") == entry.get("title") for p in pins):
                pins.append({"type": "manual", "title": entry.get("title", ""),
                             "file": entry.get("file", ""),
                             "hook": entry.get("hook", "")})
        else:
            pins = [p for p in pins if p.get("title") != entry.get("title")]

        PINS_PATH.write_text(_json.dumps(pins, indent=2))
        body = _json.dumps({"ok": True, "count": len(pins)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Path, mime: str):
        if not path.exists():
            self.send_response(404); self.end_headers(); return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _serve_operator_position(self):
        """Return operator presence + position.

        Two signals fused:
          1. Tailscale device list for `pixel-10-1` — if it has a `direct
             192.168.x.x` peer, the phone is on the LAN = operator is HOME.
             If it's `active` over relay, operator is REMOTE on tailnet.
             If `offline`, operator status unknown.
          2. ~/Documents/hark-movement/pixel10.tsv tail — when Drop-Pin pings
             land, we get lat/lon/accuracy/SSID. Distance from HOME (Palm
             Springs ~33.83/-116.55) is haversine.
        """
        import json as _json
        import math as _math
        import subprocess as _subprocess

        HOME_LAT, HOME_LON = 33.8303, -116.5453
        info = {
            "status": "unknown",
            "label": "OPERATOR · UNKNOWN",
            "via": None,
            "phone_id": None,
            "phone_addr": None,
            "drop_pin": None,
        }

        # Signal 1 — Tailscale phone direct/relay peer
        try:
            ts = _subprocess.run(
                ["/Applications/Tailscale.app/Contents/MacOS/Tailscale", "status"],
                capture_output=True, text=True, timeout=4,
            ).stdout
            for line in ts.splitlines():
                parts = line.split()
                if len(parts) < 5: continue
                tip, host = parts[0], parts[1]
                if host not in ("pixel-10-1", "maxxxnaty", "pixel-10"): continue
                rest = " ".join(parts[4:])
                if "offline" in rest:
                    continue
                info["phone_id"] = host
                # Find "direct <ip:port>" or "relay" or "idle"
                if "direct " in rest:
                    ip_part = rest.split("direct ", 1)[1].split(",")[0].split()[0]
                    info["phone_addr"] = ip_part
                    if ip_part.startswith("192.168.") or ip_part.startswith("10.") or ip_part.startswith("172."):
                        info["status"] = "home"
                        info["label"] = "OPERATOR · HOME"
                        info["via"] = "tailscale-lan"
                        break
                # relay or idle → remote on tailnet
                info["status"] = "deployed"
                info["label"] = "OPERATOR · DEPLOYED"
                info["via"] = "tailscale-relay"
        except (FileNotFoundError, _subprocess.TimeoutExpired, _subprocess.SubprocessError):
            pass

        # Signal 2 — Drop-Pin GPS (when available)
        tsv = Path.home() / "Documents" / "hark-movement" / "pixel10.tsv"
        if tsv.exists():
            try:
                lines = [l for l in tsv.read_text().splitlines() if l and not l.startswith("timestamp")]
                if lines:
                    last = lines[-1].split("\t")
                    if len(last) >= 3:
                        lat, lon = float(last[1]), float(last[2])
                        # haversine
                        R = 6371.0
                        p1, p2 = _math.radians(HOME_LAT), _math.radians(lat)
                        dp = _math.radians(lat - HOME_LAT)
                        dl = _math.radians(lon - HOME_LON)
                        a = _math.sin(dp/2)**2 + _math.cos(p1)*_math.cos(p2)*_math.sin(dl/2)**2
                        km = 2 * R * _math.asin(_math.sqrt(a))
                        mi = km * 0.621371
                        info["drop_pin"] = {
                            "ts": last[0],
                            "lat": lat,
                            "lon": lon,
                            "km_from_home": round(km, 2),
                            "mi_from_home": round(mi, 2),
                            "accuracy_m": float(last[3]) if len(last) >= 4 and last[3] else None,
                            "ssid": last[4] if len(last) >= 5 else None,
                            "battery_pct": int(last[5]) if len(last) >= 6 and last[5].isdigit() else None,
                        }
                        # If we have GPS, refine the label.
                        if mi < 0.2:
                            info["label"] = "OPERATOR · HOME"
                            info["status"] = "home"
                        else:
                            info["label"] = f"OPERATOR · {mi:.1f} MI"
                            info["status"] = "deployed"
                        info["via"] = "drop-pin-gps"
            except (OSError, ValueError, IndexError):
                pass

        body = _json.dumps(info).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_doorbell_brightness(self):
        """Serve the latest doorbell-brightness-watcher state as JSON.
        Reads ~/.local/state/doorbell-brightness.json — written every 5min
        by the Mac launchd job (~/bin/doorbell-brightness-watcher.sh)."""
        import json as _json
        state_file = Path.home() / ".local" / "state" / "doorbell-brightness.json"
        if not state_file.exists():
            body = _json.dumps({"status": "no-state", "strikes": 0, "last_brightness_pct": None}).encode()
        else:
            try:
                body = state_file.read_bytes()
            except OSError:
                body = _json.dumps({"status": "read-error"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_sentinel_digest(self):
        """Parse the Sentinel digest tail (synced from Pixel 6) into JSON.

        Each digest line: `[YYYY-MM-DD HH:MM:SS] [level] watcher: message`
        Returns: {"last_synced": str|null, "entries": [{ts,level,watcher,message}]}
        """
        import json as _json
        import re as _re
        payload = {"last_synced": None, "entries": []}
        if SENTINEL_DIGEST_SYNCED.exists():
            try:
                payload["last_synced"] = SENTINEL_DIGEST_SYNCED.read_text().strip() or None
            except OSError:
                pass
        if SENTINEL_DIGEST_TAIL.exists():
            try:
                lines = SENTINEL_DIGEST_TAIL.read_text().splitlines()
            except OSError:
                lines = []
            pat = _re.compile(r"^(\S+ \S+) \[(\w+)\] ([^:]+): (.*)$")
            for line in lines:
                m = pat.match(line.strip())
                if not m:
                    continue
                payload["entries"].append({
                    "ts": m.group(1),
                    "level": m.group(2),
                    "watcher": m.group(3),
                    "message": m.group(4),
                })
        # Newest first for the UI
        payload["entries"].reverse()
        body = _json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_noaa(self, rel_path: str):
        """Serve a file from ~/Documents/noaa-captures/. Read-only. Bare
        /noaa/ returns a JSON directory listing so the dashboard can build a
        gallery; /noaa/<sat-folder>/ returns the per-pass file listing;
        /noaa/<sat-folder>/<file>.png streams the image with the right MIME.
        Path-traversal hardened via Path.resolve() + is_relative_to check.
        """
        import json as _json
        # Strip query string and trailing slashes
        clean = rel_path.split("?", 1)[0].strip("/")

        # /noaa/  → list pass directories newest-first
        if clean == "":
            if not NOAA_CAPTURE_ROOT.exists():
                payload = _json.dumps({"passes": []}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(payload)
                return
            dirs = sorted(
                (d for d in NOAA_CAPTURE_ROOT.iterdir() if d.is_dir()),
                key=lambda d: d.stat().st_mtime,
                reverse=True,
            )
            entries = []
            for d in dirs:
                images = sorted(p.name for p in d.glob("*.png"))
                entries.append({
                    "dir": d.name,
                    "captured_at": d.stat().st_mtime,
                    "images": images,
                })
            payload = _json.dumps({"passes": entries}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(payload)
            return

        target = (NOAA_CAPTURE_ROOT / clean).resolve()
        # Block any escape attempt (../, symlink-to-outside, etc.)
        try:
            target.relative_to(NOAA_CAPTURE_ROOT)
        except ValueError:
            self.send_response(403); self.end_headers(); return

        if target.is_dir():
            # /noaa/<dir>/  → JSON file listing
            images = sorted(p.name for p in target.glob("*.png"))
            payload = _json.dumps({"dir": target.name, "images": images}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(payload)
            return

        if not target.is_file():
            self.send_response(404); self.end_headers(); return

        # Only allow PNG/JSON/CBOR — same files satdump writes
        mime = {
            ".png": "image/png",
            ".json": "application/json",
            ".cbor": "application/cbor",
            ".wav": "audio/wav",
        }.get(target.suffix.lower())
        if mime is None:
            self.send_response(403); self.end_headers(); return

        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _proxy_post(self, base_url: str, path: str, method: str = "POST"):
        target = base_url + path
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length) if length else b""
        ct     = self.headers.get("Content-Type", "application/json")
        # scan endpoints can take longer (EasyOCR / face model init)
        timeout = 90 if "/scan/" in path else 15
        try:
            req = urllib.request.Request(target, data=body, method=method)
            req.add_header("Content-Type", ct)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                for key, val in resp.headers.items():
                    if key.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(key, val)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(resp_body)
        except urllib.error.URLError:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b'{"error":"backend offline"}')
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(str(e).encode())

    def _proxy_websocket(self, path: str):
        try:
            backend = socket.create_connection(("localhost", 1984), timeout=10)
        except OSError:
            self.send_response(503); self.end_headers(); return

        request = f"GET {path} HTTP/1.1\r\nHost: localhost:1984\r\n"
        for key, val in self.headers.items():
            if key.lower() != "host":
                request += f"{key}: {val}\r\n"
        request += "\r\n"
        backend.sendall(request.encode())

        client = self.connection
        self.close_connection = True

        def pipe(src, dst):
            try:
                while True:
                    data = src.recv(4096)
                    if not data:
                        break
                    dst.sendall(data)
            except OSError:
                pass
            finally:
                try: dst.shutdown(socket.SHUT_WR)
                except OSError: pass

        t = threading.Thread(target=pipe, args=(backend, client), daemon=True)
        t.start()
        pipe(client, backend)
        t.join(timeout=5)
        backend.close()

    def _proxy(self, base_url: str, path: str):
        target = base_url + path
        # Endpoints that hit a local LLM can take 60-180s on cold start
        # (qwen3:14b is a 14B thinking model). Bump their timeout.
        timeout = 180 if ("/intel/morning-brief" in path or "/agent/query" in path or "/scan/" in path) else 10
        try:
            req = urllib.request.Request(target)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                self.send_response(resp.status)
                for key, val in resp.headers.items():
                    if key.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(key, val)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
        except urllib.error.URLError:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b'{"error":"backend offline"}')
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(str(e).encode())


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", SERVE_PORT), DashboardHandler)
    print(f"COMMAND CENTER dashboard → http://0.0.0.0:{SERVE_PORT}", flush=True)
    server.serve_forever()
