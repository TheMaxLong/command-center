# COMMAND CENTER

## Overview
COMMAND CENTER is a home security AI surveillance system ("Situational Awareness System"). It provides a dark-themed tactical dashboard for monitoring Tapo IP cameras (D210/D230/C310), generic RTSP cameras, and go2rtc streams. Features include motion detection, YOLOv8 AI detection, multi-person profiling, cross-camera tracking, Kalman-filter ByteTrack multi-object tracking, license plate recognition, real-time external intelligence feeds, gait biometric analysis, FBI face intelligence cross-reference, pattern-of-life behavioral modeling, entity relationship graph, threat scoring, and predictive arrival forecasting.

## Architecture

### Frontend
- **`dashboard/index.html`** — Single-file static dashboard (HTML/CSS/JS). Dark tactical UI with JetBrains Mono font, real-time camera tiles, AI detection readouts with direction/dwell badges, scene summary ticker, event log, person profiles (custom rename + cross-camera timeline), 5-week heatmap, recurring schedule, pinned anomalies, Intel Briefing, Alerts, PALANTIR terminal overlay, and MONITOR MODE fullscreen.
- Served via **`serve_dashboard.py`** on port **5000** (Replit webview port)
- Uses relative URLs (`/api/*` and `/go2rtc/*`) proxied by the dashboard server
- **DO NOT MODIFY** dashboard/index.html links or layout

### Backend — Core
- **`camera_watcher.py`** — Multi-camera watcher v2. Supports Tapo, RTSP, go2rtc. Wires AI + profiling + Kalman tracker + LPR + gait + face intel + pattern engine + threat scoring. HTTP API on port **8181**.
- **`ai_engine.py`** — YOLOv8 AI detection v2: multi-instance per class, per-class confidence thresholds, 15 COCO classes, 64-dim enriched embedding, annotated snapshots.
- **`profiler.py`** — Person profiler v2: 64-dim embedding, EMA rolling avg, profile dedup/merge, custom labels, cross-camera tracking.
- **`intel_engine.py`** — Intelligence engine: scene_summary, stranger_alert, daily_briefing, cross_camera_timeline, active_alerts.
- **`trend_analyzer.py`** — Trend analyzer: schedule inference, anomaly detection, velocity tracking, camera comparison.
- **`event_db.py`** — SQLite event store: WAL mode, intel_alerts table, profile merge/label.
- **`query_agent.py`** — PALANTIR AI query agent: rule-based NLU + optional LLM (OpenAI/Anthropic). 25+ intents.

### Backend — Intelligence Layer (NEW)
- **`tracker_kalman.py`** — ByteTrack-inspired multi-object tracker: 6D Kalman filter [cx,cy,vx,vy,w,h], Hungarian assignment, TENTATIVE→CONFIRMED→LOST lifecycle, velocity-derived direction.
- **`lpr_engine.py`** — License plate recognition: OpenCV contour + EasyOCR, plate log, watchlist.
- **`intel_feeds.py`** — Real-time external intelligence: NWS weather alerts, USGS seismic (200km), CAL FIRE incidents, Citizen 911 incidents, threat level scoring. Background refresh every 3 min.
- **`gait_engine.py`** — Gait biometric analysis: YOLOv8-pose 17-keypoint skeleton, 18-dim normalized gait vector (stride width, torso lean, arm swing, hip sway, step height, knee/elbow angles, shoulder-hip ratio), cosine similarity matching, per-track Kalman-smoothed signature accumulation.
- **`face_intel.py`** — FBI face intelligence: downloads FBI Most Wanted (1,160+ records) for LA/San Diego/Las Vegas field offices + national categories. OpenCV DNN face detector (ResNet SSD), 80-dim feature vector (RGB hist + YCbCr skin tone + gradient), cosine similarity matching, operator-managed POI database, background refresh every 6h.
- **`pattern_engine.py`** — Pattern-of-Life & Entity Relationship Engine: per-entity behavioral models (hour/day distributions, dwell time, camera frequency), predictive arrival forecasting, real-time deviation scoring, entity relationship graph (co-appearance within 5-min windows), multi-camera movement chain reconstruction, threat scoring with multiple factors.
- **`traffic_cam.py`** — Neighborhood overwatch: stitches OpenStreetMap tiles into a live tactical map view of San Rafael Ave & Palm Canyon Dr. Tactical dark overlay, crosshair, GPS labels. 10-min refresh. Supports live MJPEG override via NEIGHBORHOOD_CAM_URL.

### Backend — Universal Camera Framework (NEW)
- **`camera_adapters.py`** — Multi-manufacturer adapter registry. 14 vendors: tapo, rtsp, mjpeg, http_snap, hikvision, amcrest/dahua, reolink, wyze, onvif, usb, bluetooth, ring/arlo/nest stubs. Subclass `CameraAdapter` + call `register_adapter()` to add new vendors. Each adapter implements `snapshot()`, `stream_url()`, `capabilities()`. The `adapter_poll_loop` in camera_watcher.py drives any registered adapter through the full AI pipeline.
- **`camera_discover.py`** — Network camera auto-discovery. Layers: ONVIF WS-Discovery (UDP multicast 239.255.255.250:3702), parallel TCP port scan of common ports (80/554/8554/8800/8000/37777), vendor fingerprinting via characteristic URL probes. Returns CameraInfo objects + ready-to-paste YAML block. Routes: `/api/discover`, `/api/discover/yaml`, `/api/adapters`.

### Backend — Palantir Forward Intelligence (NEW)
- **`entity_resolution.py`** — Cross-modal identity fusion. Combines face (40%), gait (30%), appearance (15%), spatial-temporal (10%), and existing profile aliases (5%) into a single weighted score. Above 85% reinforces existing entity, 70-85% logs a merge suggestion, below 70% spawns new entity. Tracks merge_log for human review. Manual `merge_entities()` and `unmerge_entity()` operations. SQLite store at `/tmp/palm_entities.db`. Routes: `/intel/entities`, `/intel/entity/<id>`, `/intel/merge_log`.
- **`forward_intel.py`** — Predictive threat scenarios + behavior classification. Classifies entities as: visitor, regular, scout, loiterer, runner, lookout, intruder, occasional. Scenario builders: scouting (hit 3+ cameras in 15min), convergence (3+ entities at one camera in 5min window), anomalous absence (regulars overdue by 1.6×), loitering, intruder. Routes: `/intel/forecast`, `/intel/behavior`, `/intel/classify/<id>`.

### Proxy Server
- **`serve_dashboard.py`** — Serves `dashboard/` static files on port 5000, proxies `/api/*` → 8181, `/go2rtc/*` → 1984.

### Configuration
- **`cameras.yaml`** — Camera configuration. `type:` field maps to a registered adapter (see camera_adapters.py); built-ins: tapo, rtsp, go2rtc; universal: hikvision, reolink, amcrest, dahua, onvif, mjpeg, http_snap, wyze, usb, bluetooth, ring, arlo, nest.
- Key env vars: `TAPO_IP`, `TAPO_PASSWORD`, `DB_PATH`, `ENTITY_RESOLUTION_DB`, `HOME_LAT`, `HOME_LON`, `HOME_ZIP`, `HOME_NAME`, `NEIGHBORHOOD_CAM_URL`, `FBI_FIELD_OFFICES`, `GAIT_MODEL`, `FACE_MATCH_THRESH`, `BT_ENABLE`

### Backend — Hardened Operations Layer (NEW)
- **`notifier.py`** — Push notification engine. Delivers alerts to phone/devices via ntfy.sh (no account needed — just set `NTFY_TOPIC`) with optional Twilio SMS fallback. Priority mapping: critical→5 (breaks Do Not Disturb), high→4, medium→3, low→2. Rate-limited per (type, camera) pair with configurable cooldown (default 5min). Fire-and-forget via background thread. Wired into: FBI face match (critical), threat score RED/ORANGE (critical/high), stranger alert (medium). Routes: `/api/notify/status`, `/api/notify/test`. PALANTIR: *"notification status"*, *"test notification"*.
- **`evidence_export.py`** — Evidence package generator. One call bundles: `manifest.json`, `report.txt` (human-readable incident report), `timeline.json` (chronological sightings), `entity_profile.json` (fusion data), `pattern_of_life.json`, `face_matches.json`, `gait_data.json`, `snapshots/` (annotated JPEGs per camera). Returns in-memory ZIP bytes. Routes: `/api/evidence/<entity_id>?hours=72`, `/api/evidence/profile/<id>?hours=72` → `Content-Disposition: attachment` ZIP download. PALANTIR: *"generate evidence for profile 3"*, *"evidence for E0012345678"*.
- **API Authentication** — `PALM_API_TOKEN` env var gates all external requests to the watcher API (:8181). Localhost/127.0.0.1 (the dashboard proxy) always passes through unauthenticated. External callers must include `X-Palm-Token: <token>` header. If `PALM_API_TOKEN` is not set, auth is disabled (dev mode). 401 returns JSON error with hint.

### Extending for Claude Code
- **Add a camera vendor**: subclass `CameraAdapter` in camera_adapters.py, implement `snapshot()` + `stream_url()` + `capabilities()`, call `register_adapter("vendor_name", YourClass)`. Use HikvisionAdapter or ReolinkAdapter as templates. The `adapter_poll_loop` picks it up automatically.
- **Add a discovery layer**: implement `discover_<name>()` in camera_discover.py and call from `discover_all()`. Add fingerprint patterns to VENDOR_PROBES.
- **Add a fusion modality** (entity resolution): implement `_score_<modality>()` and add weight to FUSION_WEIGHTS in entity_resolution.py.
- **Add a forward-intel scenario**: implement `_scenario_<name>()` returning a list of scenario dicts and append to the tuple in `build_scenarios()`.
- **Add a query intent**: append to `_INTENT_PATTERNS` (Palantir patterns FIRST, generic LAST) in query_agent.py and add the corresponding `_handle_<intent>()` function. Register in the `handlers` dict in `query()`.

### Mobile PWA — Pixel 10 Operator App (NEW)
- **`dashboard/mobile.html`** — Palantir-style progressive web app optimized for Android phones. Installable to home screen ("Add to Home Screen" in Chrome). Five tactical tabs:
  - **FEED** — Live annotated camera snapshots (polling snap_ann every 2.5s), canvas overlay for additional AI bounding boxes, **17-point COCO skeleton** rendered in real-time over person detections with per-joint color coding, gait radar chart (hexagonal, 6 axes) for each detected person, per-camera switching, threat level chip strip.
  - **INTEL** — Entity cards with profile thumbnails, sighting counts, camera trail, behavior class badge, embedded gait radar per entity. Forward intelligence scenario cards with probability bars. System-wide stat overview (profiles, entities, events today, alerts).
  - **ALERTS** — Chronological intel_alerts feed, severity color-coded (red/orange/yellow/blue), floating banner at top for new critical/high alerts.
  - **TERMINAL** — Full PALANTIR query interface with quick-tap command chips (briefing, strangers, forecast, who today, threat, entities, notifications, help), monospace dark terminal output, full keyboard input.
  - **OVERWATCH** — Neighborhood traffic cam map (Palm Springs) + area intelligence briefing.
- **`dashboard/manifest.json`** — PWA manifest for "Add to Home Screen" installability.
- **`dashboard/sw.js`** — Service worker: network-first for API calls, cache-first for app shell.
- Route: `/mobile` served by `serve_dashboard.py`.

## Workflows
- **Start application** — `python3.12 serve_dashboard.py` on port 5000

## Camera Watcher API (port 8181)
```
GET  /status                      → all cameras status JSON
GET  /status/<cam_id>             → one camera
GET  /clip/<cam_id>               → latest video clip (mp4)
GET  /snap/<cam_id>               → latest snapshot (jpeg)
GET  /snap_ann/<cam_id>           → AI-annotated snapshot (jpeg)
GET  /events[?camera=&limit=]     → recent events from DB
GET  /trends[?camera=&weeks=]     → full trend report
GET  /profiles                    → person profiles summary
GET  /thumb/<profile_id>          → profile thumbnail
GET  /profile/<id>/timeline       → cross-camera sightings timeline
PATCH /profile/<id>/label         → set custom name
GET  /intel/briefing[?camera=]    → 24h plain-English briefing
GET  /intel/alerts[?camera=]      → active anomaly + stranger alerts
GET  /intel/velocity[?camera=]    → event rate trend
GET  /intel/patterns              → pattern-of-life summary for all entities
GET  /intel/predictions           → arrival predictions for known regulars
GET  /intel/graph                 → entity relationship graph
GET  /intel/pol_briefing          → Palantir-style pattern-of-life briefing
GET  /intel/wanted[?q=]          → FBI wanted persons database (search with ?q=)
GET  /intel/match_log             → face comparison match history
GET  /intel/movement/<id>         → cross-camera movement chain for profile
GET  /feeds                       → all external intel feeds combined
GET  /feeds/earthquakes           → USGS seismic data
GET  /feeds/weather               → NWS weather alerts
GET  /feeds/fire                  → CAL FIRE incidents
GET  /feeds/crime                 → Citizen 911 incidents
GET  /feeds/plates                → LPR plate log
GET  /feeds/briefing              → external threat briefing
GET  /feeds/gait                  → gait biometric profiles
GET  /trafficcam                  → neighborhood overwatch map image (JPEG)
GET  /trafficcam/status           → traffic cam module status
POST /agent/query                 → PALANTIR AI query {"text": "...", "camera": "..."}
```

## PALANTIR Agent — Intents
```
SURVEILLANCE:   summary, who_today, stranger_check, anomaly_check, velocity,
                camera_compare, count_query, time_query, recent_events, person_info,
                watchlist_add, watchlist_show
EXTERNAL INTEL: earthquake, weather_alert, fire_intel, local_incidents, area_threat, plates
PALANTIR LAYER: wanted_persons, gait_intel, pattern_intel, predictions, threat_score
```

## AI Models
- YOLOv8s/n — object detection (person, vehicle, animals, accessories)
- YOLOv8n-pose — 17-keypoint skeleton for gait analysis
- OpenCV ResNet SSD — face detection
- All run locally, no cloud inference

## Key Thresholds
- Person match: cosine similarity ≥ 0.88
- Profile merge: ≥ 0.97
- Gait match: ≥ 0.88 cosine (strong: ≥ 0.94)
- Face match: ≥ 0.72 (flag), ≥ 0.88 (high confidence)
- Threat RED: score ≥ 0.70 | ORANGE: ≥ 0.45 | YELLOW: ≥ 0.25

## Home Location
- Palm Springs, CA — 33.8303°N, 116.5453°W, ZIP 92262
- Configurable via HOME_LAT, HOME_LON, HOME_ZIP, HOME_NAME env vars

## Dependencies
- Python 3.12
- `pyyaml`, `Pillow`, `pytapo`, `ultralytics`, `opencv-python-headless`, `numpy`, `scipy`
- System: `ffmpeg`
- Optional: `easyocr` (LPR), `openai` or `anthropic` (LLM upgrade)

## Usage Notes
- Camera watcher backend requires actual Tapo hardware to capture live footage
- Dashboard loads in standby mode without cameras (all API calls gracefully fail)
- go2rtc streams (port 1984) require separate go2rtc installation
- DB default: `/data/events.db` — override via DB_PATH env var
- FBI database is downloaded at watcher startup (~1,160 records from regional field offices)
- Gait analysis activates automatically once ≥5 frames of a track are accumulated
- Pattern-of-life models require ≥5 sightings per entity for predictions
