#!/usr/bin/env python3.12
"""
PALM COMMAND — Dashboard server for Replit.
Serves the static dashboard on port 5000 and proxies:
  /api/*     → camera watcher on :8181
  /go2rtc/*  → go2rtc on :1984
"""
import os
import urllib.request
import urllib.error
from http.server import SimpleHTTPRequestHandler, HTTPServer
from pathlib import Path

DASHBOARD_DIR = Path(__file__).parent / "dashboard"
SERVE_PORT = 8888
WATCHER_URL = "http://localhost:8181"
GO2RTC_URL = "http://localhost:1984"


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def log_message(self, fmt, *args):
        pass  # suppress per-request logs

    def do_GET(self):
        if self.path.startswith("/api/"):
            self._proxy(WATCHER_URL, self.path[4:])  # strip /api
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


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", SERVE_PORT), DashboardHandler)
    print(f"PALM COMMAND dashboard → http://0.0.0.0:{SERVE_PORT}", flush=True)
    server.serve_forever()
