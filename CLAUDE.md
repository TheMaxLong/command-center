# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

PALM COMMAND is a Palantir-style home surveillance dashboard. It runs as a Docker stack on a Mac and serves a dark tactical web UI over Tailscale for remote access. All source lives in `~/palm-command/` and is synced to this repo.

## Running the Stack

```bash
# Full start (Docker + dashboard proxy + opens browser)
cd ~/palm-command && ./start.sh

# Or manually:
docker compose up -d          # starts go2rtc + vision-watcher
python3 serve_dashboard.py    # starts proxy on :8888
```

**Ports:**
- `:8888` — Dashboard (serve_dashboard.py, the entry point for the browser)
- `:8181` — Camera watcher API (inside Docker, proxied via /api/)
- `:1984` — go2rtc (RTSP relay, proxied via /go2rtc/)

**After any Python change to vision-watcher:**
```bash
docker compose build vision-watcher && docker compose up -d vision-watcher
```

**After any change to serve_dashboard.py or dashboard/:**
```bash
# serve_dashboard.py runs via PM2
pm2 restart serve_dashboard

# Or kill/restart manually if not using PM2
pkill -f serve_dashboard.py && python3 ~/palm-command/serve_dashboard.py &
```

**SD card backfill (pull missed clips from Tapo camera):**
```bash
docker compose run --rm backfill --days 7        # last 7 days
docker compose run --rm backfill --dry-run        # preview only
docker compose run --rm backfill --list-dates     # what's on SD card
```

## Architecture

### Request Flow
```
Browser → :8888 (serve_dashboard.py)
           ├── /api/*      → proxied to :8181 (vision-watcher in Docker)
           ├── /go2rtc/*   → raw TCP tunnel to :1984 (WebSocket streams)
           └── static      → dashboard/*.html served directly
```

`serve_dashboard.py` uses `ThreadingHTTPServer` and a raw TCP WebSocket tunnel (not urllib) for go2rtc — urllib strips `Connection: Upgrade` which breaks WebRTC/MSE streams.

### Docker Volumes
- `vision-data:/data` — SQLite database (`events.db`)
- `cam-media:/tmp/cams` — rolling clips/snaps + `archive/` timestamped copies + `backfill/` SD card imports

### Core Backend Modules (camera_watcher.py wires these together)

| Module | Role |
|---|---|
| `camera_watcher.py` | Main process: asyncio poll loops per camera, HTTP API on :8181, loads cameras.yaml |
| `event_db.py` | SQLite WAL store — events, detections, profiles, alerts, manual_scans |
| `ai_engine.py` | YOLOv8 detection, annotation, crop extraction, 64-dim embeddings |
| `profiler.py` | Person re-ID: EMA embeddings, cosine match, profile merge/label |
| `tracker_kalman.py` | ByteTrack multi-object tracker (Kalman 6D state, Hungarian assignment) |
| `intel_engine.py` | Scene summaries, stranger alerts, daily briefing, cross-camera timeline |
| `pattern_engine.py` | Pattern-of-life: arrival predictions, threat scoring, movement chains |
| `entity_resolution.py` | Cross-modal identity fusion (face 40% + gait 30% + appearance 15% + spatial 10% + alias 5%) |
| `lpr_engine.py` | License plate OCR (OpenCV contour + EasyOCR), plate log, watchlist |
| `gait_engine.py` | YOLOv8-pose 17-keypoint skeleton → 18-dim gait vector, cosine match |
| `face_intel.py` | FBI wanted database (LA/SD/LV field offices), OpenCV DNN face detector |
| `intel_feeds.py` | External feeds: NWS weather, USGS seismic, CAL FIRE, Citizen 911 |
| `forward_intel.py` | Predictive scenarios (scouting, convergence, loitering, intruder) |
| `backfill_tapo.py` | SD card import: lists recording dates via pytapo, downloads via HttpMediaSession |
| `query_agent.py` | PALANTIR terminal NLU — 25+ intents, rule-based + optional LLM |
| `notifier.py` | Push alerts via ntfy.sh + Twilio SMS; rate-limited per (type, cam) |
| `evidence_export.py` | Bundles annotated JPEGs + JSON timeline into a ZIP for download |
| `camera_adapters.py` | 14-vendor adapter registry — subclass `CameraAdapter`, call `register_adapter()` |
| `camera_discover.py` | Network scan: ONVIF WS-Discovery + TCP port scan + vendor fingerprinting |

### Camera Poll Loops (camera_watcher.py)
Each camera type gets its own asyncio task:
- **`tapo_poll_loop`** — TCP probe every N seconds; on motion → `HttpMediaSession` stream → ffmpeg → AI
- **`go2rtc_poll_loop`** — HTTP frame grab from go2rtc API every N seconds → YOLOv8 AI (no clip)
- **`rtsp_poll_loop`** — OpenCV frame diff for motion → ffmpeg clip + snap → AI
- **`adapter_poll_loop`** — Snapshot-poll for any vendor in camera_adapters registry

### Dashboard Pages
- `/` — Main dashboard (`index.html`): camera tiles, AI Intel panel (Event Log, Persons, Trends, Briefing, Alerts, Storage), PALANTIR terminal overlay, sidebar
- `/monitor?cam=<id>` — Fullscreen monitor with live stream (`monitor.html`)
- `/mobile` — Android PWA (`mobile.html`): Feed/Intel/Alerts/Terminal/Overwatch tabs
- `/field` — Field Scan app (`field.html`): LPR + face scan from phone camera

### cameras.yaml Keys
```yaml
- id: doorbell
  type: tapo           # tapo | go2rtc | rtsp | or any registered adapter vendor
  ip: "${TAPO_IP}"
  password: "${TAPO_PASSWORD}"
  port: 8800
  poll_interval: 3     # seconds between TCP probes
  capture_duration: 8  # seconds of video per trigger
  cooldown: 300        # seconds before re-triggering
  exclusion_zones:     # drop detections entirely (normalized 0.0–1.0 coords)
    - label: "tree"
      x1: 0.3  y1: 0.0  x2: 0.7  y2: 0.6
  known_zones:         # keep detections, suppress alerts, label profile NEIGHBOR
    - label: "neighbor-east"
      x1: 0.0  y1: 0.0  x2: 0.25  y2: 1.0
```

### Key Thresholds (tunable via env vars)
- `AI_MIN_CONF=0.55` — YOLOv8 detection confidence gate
- Person re-ID match: cosine ≥ 0.88 | merge: ≥ 0.97
- Gait match: ≥ 0.88 | strong: ≥ 0.94
- Face match: ≥ 0.72 flag | ≥ 0.88 high confidence
- Threat RED ≥ 0.70 | ORANGE ≥ 0.45 | YELLOW ≥ 0.25

### AI Pipeline (per motion event)
```
motion trigger
  → _tapo_capture() / frame grab
  → ffmpeg → clip.mp4 + snap.jpg  (rolling, overwritten each event)
  → _archive_media()              (timestamped copy to archive/)
  → ai_engine.detect()            (YOLOv8)
  → _filter_exclusions()          (drop zones)
  → _tag_known_zones()            (soft whitelist)
  → tracker_kalman.update()       (ByteTrack)
  → lpr_engine.process_snapshot() (vehicles only)
  → event_db.insert_event()
  → profiler.match_or_create()
  → gait_engine.process_frame()
  → face_intel.compare_detection()
  → entity_resolution.observe()
  → pattern_engine.score_appearance()
  → notifier (if RED/ORANGE or face hit)
```

### Adding to the System
- **New camera vendor**: subclass `CameraAdapter` in `camera_adapters.py`, implement `snapshot()` / `stream_url()` / `capabilities()`, call `register_adapter("vendor", Class)`. The `adapter_poll_loop` picks it up automatically.
- **New API endpoint**: add route to `do_GET` or `do_POST` in `Handler` class in `camera_watcher.py`, then proxy it in `serve_dashboard.py` if needed (usually not — `/api/*` is already proxied).
- **New PALANTIR intent**: add pattern to `_INTENT_PATTERNS` in `query_agent.py` and a matching `_handle_<intent>()` function, then register in the `handlers` dict in `query()`.
- **New dashboard tab in AI INTEL**: add a `<button class="ai-tab">` in `index.html`, handle it in `switchAITab()`, add a `load<Tab>()` function that populates `#ai-body`.

### Environment Variables
```
TAPO_IP              Doorbell camera IP
TAPO_PASSWORD        Tapo camera password
DB_PATH              SQLite path (default /data/events.db inside Docker)
CAMERAS_CONFIG       Path to cameras.yaml (default /config/cameras.yaml)
WATCHER_PORT         Vision-watcher HTTP port (default 8181)
AI_MIN_CONF          YOLOv8 confidence floor (default 0.35, currently 0.55)
NTFY_TOPIC           ntfy.sh topic for push notifications
PALM_API_TOKEN       Gates external requests to :8181 (localhost always bypasses)
HOME_LAT/LON/ZIP     Location for weather/seismic feeds (Palm Springs CA)
ARCHIVE_RETENTION_DAYS  Auto-purge window (default 14)
```
