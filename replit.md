# PALM COMMAND

## Overview
PALM COMMAND is a home security camera monitoring system ("Situational Awareness System"). It provides a dark-themed tactical dashboard for monitoring Tapo IP cameras (D210/D230/C310), generic RTSP cameras, and go2rtc streams. Features include motion detection, YOLOv8 AI detection, multi-person profiling, cross-camera tracking, event logging, trend analysis, and an intelligence engine that generates plain-English scene summaries and alerts.

## Architecture

### Frontend
- **`dashboard/index.html`** — Single-file static dashboard (HTML/CSS/JS). Dark tactical UI with JetBrains Mono font, real-time camera tiles, AI detection readouts, scene summary ticker, event log, person profiles (with custom rename + cross-camera timeline), 5-week activity heatmap, recurring schedule, pinned anomalies, Intel Briefing, and Alerts panels.
- Served via **`serve_dashboard.py`** on port **5000** (Replit webview port)
- Uses relative URLs (`/api/*` and `/go2rtc/*`) which are proxied by the dashboard server

### Backend
- **`camera_watcher.py`** — Multi-camera watcher v2. Supports Tapo, RTSP, and go2rtc camera types. Wires AI + profiling + intel engine. HTTP API on port **8181**.
- **`ai_engine.py`** — YOLOv8 AI detection v2: multi-instance per class, per-class confidence thresholds, expanded COCO classes (dog/cat/bird/backpack/suitcase/laptop/phone), 64-dim enriched embedding, bounding box annotation with numbered multi-instance labels.
- **`profiler.py`** — Person profiler v2: 64-dim embedding (colour+spatial+edge+hue), EMA rolling average, profile dedup/merge (MERGE_THRESHOLD=0.97), custom name labels, `first_seen_today`/`returning_today` flags, cross-camera tracking.
- **`intel_engine.py`** — Intelligence engine: `scene_summary()`, `stranger_alert()`, `daily_briefing()`, `cross_camera_timeline()`, `active_alerts()`, `event_velocity()`.
- **`trend_analyzer.py`** — Trend analyzer v2: schedule inference (mean+1σ), anomaly detection (z>2σ), velocity tracking (this week vs last week), camera comparison, pinned anomalies.
- **`event_db.py`** — SQLite event store v2: WAL mode, `intel_alerts` table, `merge_profiles()`, `set_profile_label()`, `get_events_in_range()`.
- **`backfill_tapo.py`** — One-shot script to backfill historical events from Tapo SD card.

### Proxy Server
- **`serve_dashboard.py`** — Python HTTP server that:
  - Serves `dashboard/` as static files on port 5000
  - Proxies `/api/*` → `localhost:8181` (camera watcher API)
  - Proxies `/go2rtc/*` → `localhost:1984` (go2rtc streams)

### Configuration
- **`cameras.yaml`** — Camera configuration (copy from `cameras.yaml.example`)
- Key env vars: `TAPO_IP`, `TAPO_PASSWORD`, `DB_PATH`, `CAMERAS_CONFIG`, `WATCHER_PORT`, `AI_MODEL`, `AI_MIN_CONF`, `PROFILE_MATCH_THRESH`, `PROFILE_MERGE_THRESH`, `PROFILE_MIN_SIGHTINGS`

## Workflows
- **Start application** — Runs `python3.12 serve_dashboard.py` on port 5000 (webview)

## Camera Watcher API (port 8181)
```
GET  /status                    → all cameras status JSON
GET  /status/<cam_id>           → one camera (includes detections, profiles, summary)
GET  /clip/<cam_id>             → latest video clip (mp4)
GET  /snap/<cam_id>             → latest snapshot (jpeg)
GET  /snap_ann/<cam_id>         → AI-annotated snapshot (jpeg, numbered multi-instance boxes)
GET  /events[?camera=&limit=]   → recent events from DB
GET  /trends[?camera=&weeks=]   → full trend report: heatmap + schedule + anomalies + velocity + camera_comparison
GET  /profiles                  → person profiles summary (with habits, badges, visits/7d)
GET  /thumb/<profile_id>        → profile thumbnail (jpeg)
GET  /profile/<id>/timeline     → cross-camera sightings timeline
PATCH /profile/<id>/label       → set custom name {"label": "..."}
GET  /intel/briefing[?camera=]  → 24h plain-English briefing
GET  /intel/alerts[?camera=]    → active anomaly + stranger alerts
GET  /intel/velocity[?camera=]  → event rate trend (this week vs last)
```

## AI Engine (v2)
- Model: YOLOv8s with automatic fallback to YOLOv8n
- All instances per class returned (no more single-detection-per-class bug)
- Per-class confidence thresholds: person=0.42, vehicles=0.30-0.35, animals=0.40
- 15 COCO classes: person, bicycle, car, motorcycle, bus, truck, bird, cat, dog, backpack, umbrella, handbag, suitcase, laptop, cell phone
- 64-dim enriched embedding: 48-dim RGB histogram + 8-dim spatial split + 4-dim brightness/contrast/saturation/edge + 4-dim HSV hue zones

## Person Profiler (v2)
- MATCH_THRESHOLD=0.88 (cosine similarity to re-identify known profile)
- MERGE_THRESHOLD=0.97 (auto-merge near-duplicate profiles)
- MIN_SIGHTINGS=4 (sightings before REGULAR label)
- EMA alpha=0.10 (slow adaptation to appearance changes)
- Returns `returning_today` and `first_seen_today` flags per event

## Dashboard Panels
- **EVENT LOG** — recent events with detection tags and snapshot thumbnails
- **PERSONS** — profile cards with returning/new badges, habits, custom rename, timeline button
- **5-WEEK TRENDS** — heatmap + velocity stats + recurring schedule + pinned anomalies + camera comparison
- **BRIEFING** — 24h plain-English intelligence summary with person list and anomaly pins
- **ALERTS** — active anomaly and stranger alerts, severity-coded (HIGH/WARN/INFO)
- **Footer** — TREND velocity indicator (▲/▼/─ with week-over-week %)
- **Camera tiles** — scene summary ticker below detection tags (e.g. "2 people (1 regular, 1 unknown) · 1 vehicle")

## Dependencies
- Python 3.12
- `pyyaml`, `Pillow`, `pytapo`, `ultralytics`, `opencv-python-headless`, `numpy`
- System: `ffmpeg`

## Usage Notes
- The camera watcher backend (`camera_watcher.py`) requires actual camera hardware (Tapo IP/password) to connect
- Without cameras configured, the dashboard loads and displays the UI in standby mode (all API calls silently fail)
- go2rtc streams (port 1984) require Docker or a separate go2rtc installation
- The original project used Docker Compose; on Replit it runs natively
- DB default path: `/data/events.db` — override via `DB_PATH` env var
