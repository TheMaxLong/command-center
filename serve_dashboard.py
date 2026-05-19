#!/usr/bin/env python3.12
"""
PALM COMMAND — Dashboard server for Replit.
Serves the static dashboard on port 5000 and proxies:
  /api/*     → camera watcher on :8181
  /go2rtc/*  → go2rtc on :1984
  /mocap/*   → FreeMoCap worker
"""
import os
import json
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

# Import FreeMoCap worker (will start when first request arrives)
try:
    from freemocap_worker import get_worker
    MOCAP_WORKER = None  # Lazy init
except ImportError:
    MOCAP_WORKER = None
    print("[WARNING] freemocap_worker not available", flush=True)


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
        if self.path.startswith("/api/"):
            self._proxy(WATCHER_URL, self.path[4:])  # strip /api
        elif self.path.startswith("/archived/"):
            self._proxy(WATCHER_URL, self.path)       # archived snap/clip media
        elif self.path.startswith("/go2rtc/"):
            self._proxy(GO2RTC_URL, self.path[7:])   # strip /go2rtc
        elif self.path.startswith("/mocap/status/"):
            self._handle_mocap_status(self.path.replace("/mocap/status/", ""))
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
        if self.path.startswith("/api/"):
            self._proxy_post(WATCHER_URL, self.path[4:])
        elif self.path == "/mocap/upload":
            self._handle_mocap_upload()
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
        try:
            req = urllib.request.Request(target)
            with urllib.request.urlopen(req, timeout=10) as resp:
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

    def _handle_mocap_upload(self):
        """Handle POST /mocap/upload with a multipart file upload."""
        global MOCAP_WORKER
        if MOCAP_WORKER is None and get_worker is not None:
            MOCAP_WORKER = get_worker()
            MOCAP_WORKER.start()

        content_length = int(self.headers.get("Content-Length", 0))
        if not content_length:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"no file"}')
            return

        # Read multipart body (simplified: assumes single file with boundary)
        body = self.rfile.read(content_length)

        try:
            # Extract filename from Content-Disposition header
            disposition = self.headers.get("Content-Disposition", "")
            filename = "upload.mp4"
            if 'filename=' in disposition:
                parts = disposition.split('filename=')
                if len(parts) > 1:
                    filename = parts[1].strip('"').split('\n')[0]

            # Write to mocap-in/
            from pathlib import Path
            mocap_in = Path("/Volumes/Seagate Portable Drive/command-center/mocap-in")
            mocap_in.mkdir(parents=True, exist_ok=True)
            input_path = mocap_in / filename

            with open(input_path, "wb") as f:
                f.write(body)

            # Queue job
            job_id = MOCAP_WORKER.submit_job(input_path, source="drag-drop")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "job_id": job_id,
                "status": "queued",
                "file": str(input_path)
            }).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _handle_mocap_status(self, job_id: str):
        """Handle GET /mocap/status/{job_id}."""
        global MOCAP_WORKER
        if MOCAP_WORKER is None and get_worker is not None:
            MOCAP_WORKER = get_worker()

        if MOCAP_WORKER is None:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"mocap worker unavailable"}')
            return

        status = MOCAP_WORKER.get_job_status(job_id)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(status).encode())


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", SERVE_PORT), DashboardHandler)
    print(f"PALM COMMAND dashboard → http://0.0.0.0:{SERVE_PORT}", flush=True)
    server.serve_forever()
