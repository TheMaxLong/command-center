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
SERVE_PORT = 5000
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
        else:
            super().do_GET()

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
