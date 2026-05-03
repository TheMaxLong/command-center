# PALM COMMAND

## Overview
PALM COMMAND is a home security camera monitoring system ("Situational Awareness System"). It provides a dark-themed tactical dashboard for monitoring Tapo IP cameras (D210/D230/C310), generic RTSP cameras, and go2rtc streams. Features include motion detection, YOLOv8 AI object detection, person profiling, event logging, and trend analysis.

## Architecture

### Frontend
- **`dashboard/index.html`** — Single-file static dashboard (HTML/CSS/JS, ~1445 lines). Dark tactical UI with JetBrains Mono font, real-time camera tiles, AI detection readouts, event log, person profiles, and activity heatmap.
- Served via **`serve_dashboard.py`** on port **5000** (Replit webview port)
- Uses relative URLs (`/api/*` and `/go2rtc/*`) which are proxied by the dashboard server

### Backend
- **`camera_watcher.py`** — Main multi-camera watcher. Supports Tapo, RTSP, and go2rtc camera types. HTTP API on port **8181**.
- **`doorbell_watcher.py`** — Legacy single-camera doorbell watcher (superseded by camera_watcher.py)
- **`ai_engine.py`** — YOLOv8 AI detection (ultralytics), bounding box annotation, person crop extraction
- **`event_db.py`** — SQLite event/detection/profile storage (default path: `/data/events.db`)
- **`profiler.py`** — Person re-identification and profile management
- **`trend_analyzer.py`** — Heatmap and detection trend analysis
- **`backfill_tapo.py`** — One-shot script to backfill historical events from Tapo SD card

### Proxy Server
- **`serve_dashboard.py`** — Python HTTP server that:
  - Serves `dashboard/` as static files on port 5000
  - Proxies `/api/*` → `localhost:8181` (camera watcher API)
  - Proxies `/go2rtc/*` → `localhost:1984` (go2rtc streams)

### Configuration
- **`cameras.yaml`** — Camera configuration (copy from `cameras.yaml.example`)
- **`.env`** — Environment variables (copy from `.env.example`)
- Key env vars: `TAPO_IP`, `TAPO_PASSWORD`, `DB_PATH`, `CAMERAS_CONFIG`, `WATCHER_PORT`

## Workflows
- **Start application** — Runs `python3.12 serve_dashboard.py` on port 5000 (webview)

## Camera Watcher API (port 8181)
- `GET /status` — All cameras status JSON
- `GET /status/<cam_id>` — Single camera status
- `GET /clip/<cam_id>` — Latest video clip (mp4)
- `GET /snap/<cam_id>` — Latest snapshot (jpeg)
- `GET /snap_ann/<cam_id>` — AI-annotated snapshot (jpeg)
- `GET /events[?camera=&limit=]` — Recent events
- `GET /trends[?camera=&weeks=]` — Activity heatmap + detection summary
- `GET /profiles` — Person profiles
- `GET /thumb/<profile_id>` — Profile thumbnail

## Dependencies
- Python 3.12
- `pyyaml`, `Pillow`, `pytapo`, `ultralytics`, `opencv-python-headless`, `numpy`
- System: `ffmpeg`

## Usage Notes
- The camera watcher backend (`camera_watcher.py`) requires actual camera hardware (Tapo IP/password) to connect
- Without cameras configured, the dashboard still loads and displays the UI in standby mode
- The original project used Docker Compose; on Replit it runs natively
- go2rtc streams (port 1984) require Docker or a separate go2rtc installation
