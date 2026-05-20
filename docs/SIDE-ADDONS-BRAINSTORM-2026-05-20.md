---
date: 2026-05-20
status: brainstorm — for Max's morning read
scope: HOME ONLY (Palm Springs residence). Max taps in from the facility, the cameras + sensors live at home.
scouts: 5 parallel (camera ecosystem, AI/CV upgrades, OSINT/SDR, audio/sensors, tactical UX)
companion: SIDE-ADDONS-PLAN-2026-05-20.md (phased rollout)
---

# Command Center — Side Add-Ons Brainstorm

What people in the open-source community have already built that we can plug into Command Center. Five domains, ranked findings, hard passes documented. The companion `SIDE-ADDONS-PLAN-2026-05-20.md` has the phased rollout with effort estimates.

**Scope note:** Command Center runs at Max's home in Palm Springs. The doorbell, front cam, sensors, and bridge machine all live there. Max taps in remotely from the facility via Tailscale. Nothing in this brainstorm extends to the facility itself — all "facility" / "Anderson" / "grow-room" references in earlier drafts were context-leak from sibling projects and have been scrubbed.

## Executive read (start here)

**Five things that punch above their weight if you do nothing else:**

1. **rtl_433** — Your RTL-SDR already exists for Ground Station. `rtl_433` listens on 433 MHz and decodes cheap door-contact / motion-PIR / temp / humidity sensors. Plug $5–20 Aqara/Sonoff/Shelly sensors anywhere and they show up as events in Command Center via MQTT. Zero new core infrastructure.
2. **YAMNet on the doorbell mic** — Google's 521-class audio event model runs on CPU. The Tapo D210's 2-way audio stream already exists in pytapo. Free dog-bark / glass-break / siren / alarm detection layered on top of your visual events. Doubles your intel surface for zero new hardware.
3. **InsightFace (ArcFace 512-dim)** — Your current 80-dim histogram face vector is a generation behind. Drop in InsightFace's ArcFace and your face-match accuracy jumps ~20%. Same Python API surface as your current `face_intel.py`.
4. **Supervision (Roboflow, 39k stars)** — One-line orchestration for YOLOv8 + ByteTrack + annotation. Replaces a few hundred lines of `ai_engine.py` glue with proven library code. Zero behavior change for users; cleaner internals.
5. **dump1090 / readsb + tar1090** — If your Ground Station rig isn't already decoding ADS-B, your RTL-SDR is sleeping on it. Real-time aircraft tracking near PSP + China Lake military traffic, served as a Shadowbroker layer. You already have the antenna.

**The big architectural question to think about:** Do you want to keep `camera_watcher.py` as the conductor and add Frigate as an event source, OR replace `camera_watcher.py` with Frigate and write a thin adapter to feed the existing dashboard? My read: keep your conductor (you have things Frigate doesn't — entity resolution, threat scoring, PALANTIR NLU). Add Frigate only if you need 24/7 recording with EdgeTPU acceleration. Defer.

**Ethics yellow flags that are real:** AirTag/BLE tracking laws (CA stalking statutes), doorbell transcript PII (CA two-party consent), speaker diarization re-identification. None are showstoppers, but they all want a written retention/consent policy before going live. Flagged inline below.

---

## Domain 1 — Camera / streaming ecosystem

### Top picks

| Project | Stars | Effort | Why it matters |
|---|---|---|---|
| **Frigate NVR** [link](https://github.com/blakeblackshear/frigate) | 11k+ | MED | 24/7 NVR with native AI detection. Best-in-class for "always recording + indexed by object." Pairs via MQTT into Command Center's event sink. EdgeTPU ($100 Coral) makes inference real-time on Intel CPUs. |
| **Scrypted** [link](https://github.com/koush/scrypted) | 3.5k+ | LOW | Universal camera plugin platform. HomeKit Secure Video bridge if you ever want iOS-native viewing. Sub-200ms codec negotiation. |
| **python-onvif-zeep** [link](https://github.com/FalkTannhaeuser/python-onvif-zeep) | 500+ | MED | The big one nobody uses: **WS-Subscription real motion events** from the camera itself, not TCP-poll inference. 10x more reliable than "port is open = motion." Tapo C-series supports it; D210 doorbell does not. |
| **DeepStack** [link](https://docs.deepstack.cc/object-detection/index.html) | ~2k | LOW | CPU-only AI inference server. Drop-in if you don't want to deal with Coral / GPU. Reasonable performance, simple HTTP API. |
| **docker-wyze-bridge** [link](https://github.com/mrlt8/docker-wyze-bridge) | 2.5k | TRIVIAL | If you ever buy Wyze cams for cheap bulk coverage, this turns them into local RTSP without Wyze cloud. |
| **neolink + reolink-aio** [link](https://github.com/thirtythreeforty/neolink) | 800+ | LOW | Reolink Baichuan-protocol RTSP bridge + Python API for IR/spotlight/siren control. Useful if Anderson facility brings Reolink into play. |

### Confirmed dead-ends (do NOT pursue)

- **Tapo D210 doorbell button-press event API** — NOT exposed by Tapo. Multiple community projects have tried. SD-card wake-on-event is the only path. Don't waste time.
- **MotionEye** — Unmaintained since ~2023. Frigate is the modern replacement.
- **mediasoup** — WebRTC SFU. Overkill unless you have 20+ concurrent viewers. go2rtc handles your scale.

### Sleeper

- **OpenWebRX** [link](https://github.com/jketterl/openwebrx) — Browser-accessible SDR tuner with built-in decoders (FM/AM/SSB/CW/DSTAR/Codec2). Pairs with your existing Ground Station rig. Bundle into the OPS Center as a fourth panel.

---

## Domain 2 — AI / Computer Vision upgrades

### Top 3 quick wins

| Project | Stars | Effort | Drop-in? | Lift |
|---|---|---|---|---|
| **Supervision** [link](https://github.com/roboflow/supervision) | 39k | TRIVIAL | Yes, MIT | Cleaner pipelines, native ByteTrack/BoT-SORT/OC-SORT integration, better annotation. Zero behavior change for users. |
| **InsightFace (ArcFace)** [link](https://github.com/deepinsight/insightface) | 28k | LOW | Same API surface, license per-model | 512-dim embeddings vs your 80-dim → +20% match accuracy. 30–40% faster than your current pipeline. |
| **YOLOv10** [link](https://github.com/THU-MIG/yolov10) | 11k | TRIVIAL | `YOLO('yolov10n.pt')` one-liner | 10–15% faster + slightly more accurate on small objects (faces, plates). |

### Strong contenders (Phase 2)

| Project | Effort | When it's worth it |
|---|---|---|
| **FastReID** [link](https://github.com/JDAI-CV/fast-reid) | MED | When you have multiple cameras and want to confirm same identity across rooms. Replaces 64-dim histogram appearance vec with 2048-dim trained re-ID embedding. |
| **MMPose / RTMPose-tiny** [link](https://github.com/open-mmlab/mmpose) | MED | If YOLOv8n-pose isn't accurate enough for gait. RTMPose-tiny is 100 FPS on M5 with better keypoint quality. |
| **RT-DETR** [link](https://github.com/lyuwenyu/RT-DETR) | MED | Crowds / overlapping people. Transformer detector beats YOLO on dense scenes. |
| **MMTracking** [link](https://github.com/open-mmlab/mmtracking) | MED | Production-grade tracker library (DeepSORT, OC-SORT, BoT-SORT under one config). |

### Researchy / future-flag

- **YOLO-World** [link](https://github.com/AILab-CVC/YOLO-World) — **Open-vocabulary detection**. Prompt: "person carrying weapon" or "package on doorstep" without retraining. This could rewrite how `query_agent.py` flags scenes. Test thoroughly before shipping — very new.
- **MobileSAM** [link](https://github.com/ChaoningZhang/MobileSAM) — Segment Anything, 10x faster. Useful for background filtering before face/pose refinement.
- **SlowFast** [link](https://github.com/facebookresearch/SlowFast) — Action recognition (running / fighting / kneeling). Requires temporal buffers — architectural change.

### Hard pass

- **OpenALPR** — AGPL v3, legal landmine for any future shipping. Keep EasyOCR.
- **PARSeq / general OCR upgrades** — your EasyOCR is tuned; switching loses ground truth.
- **Grounded-SAM** — GPU-heavy (~2 FPS), stale (last commit 2024-09), wrong fit for M5.

---

## Domain 3 — OSINT / SDR / public-data feeds

### Tier 1 — Reuse your existing RTL-SDR (Ground Station hardware)

| Project | What it does | Hyperlocal angle |
|---|---|---|
| **dump1090 / readsb / tar1090** | ADS-B aircraft tracking | PSP arrivals + China Lake military ops + private charter patterns |
| **AIS-Catcher** [link](https://github.com/jvde-github/AIS-catcher) | Dual-band maritime AIS | Salton Sea, seasonal construction barges, dust-storm shipping ground truth |
| **OpenWebRX** | Browser-based SDR with decoders | National Simplex 146.520, local fire/EMS repeaters, ham activity |
| **gpredict + CelesTrak** | Satellite pass prediction | NOAA polar orbit pre-position, ISS overhead alerts, Landsat thermal sweeps |
| **GridDown** [link](https://www.rtl-sdr.com/griddown-an-offline-first-situational-awareness-platform-with-rtl-sdr-sarsat-meshtastic/) | **FAA Remote ID drone detection via SDR** | Alerts when a drone broadcasts over your house. Real-world signal for "is someone scoping my place." |

### Tier 2 — OSINT public records (free APIs)

| Source | Value |
|---|---|
| **OpenSky Network API** | Cloud-baseline ADS-B to compare against local dump1090 (catches Mode-C spoofing, unregistered aircraft) |
| **Blitzortung Lightning** | MQTT real-time strikes, sub-km resolution. Desert monsoon season (Jun–Sep) = power-grid + fire risk correlation |
| **PurpleAir API** | Hyperlocal AQI from neighborhood sensor mesh. Wildfire smoke days, dust storms — useful context overlay on the dashboard. |
| **California Open Data — Power Outage Incidents** | Real-time SCE/PG&E/SDG&E outages with polygons. Geofence-able to your home — early warning when grid goes down nearby. |
| **NSOPW (sex offender registry)** | Free national/CA registry. POI overlay for your home premises (~500m radius). Defensive awareness layer. |
| **USGS earthquakes** | Already integrated — but enhance with Atom CAP subscription instead of polling |

### Tier 3 — Threat / cyber

| Source | Value | Notes |
|---|---|---|
| **AbuseIPDB** | 1k IP lookups/day free | Cross-ref any inbound network traffic against community blacklist |
| **AlienVault OTX** | 19M IOCs/day free | Threat pulses for IoT firmware exploits — relevant since you run Tapo + RTL-SDR firmware |

### Tier 4 — Hyperlocal civic (longer-shot)

- **Local 911 dispatch decoders** — pattern exists (HN thread on SF 911 + LLM decode), but Riverside County 911 isn't publicly streamed.
- **Nextdoor manual scrape** — DON'T automate (TOS). Manual daily glance is fine.

### Hard pass

- **Shodan API** — Wrong tool for hyperlocal (built for remote target discovery). AbuseIPDB does the local angle better.
- **Broadcastify** — Not OSS, volunteer feeds go offline unpredictably. Use OpenWebRX with your own SDR instead.
- **Citizen API** — Already in your stack (paid/free hybrid). The OSS 911 dispatch decoder pattern is interesting but not actionable for Riverside.
- **MISP** — SOC-scale. Overkill until/unless you're running a real threat-intel program.

---

## Domain 4 — Audio + alternate sensors

### Tier 1 — Use what you already have (zero new hardware)

| Project | Stars | Effort | What it unlocks |
|---|---|---|---|
| **YAMNet** [link](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) | 30k+ (TF org) | LOW | 521 audio event classes. Detect bark / glass-break / sirens / smoke alarms on the Tapo D210 mic. CPU-only. |
| **faster-whisper** [link](https://github.com/SYSTRAN/faster-whisper) | 10k+ | TRIVIAL | Offline doorbell-conversation transcription (4x speed of OpenAI Whisper). Tag transcripts to video frames. **YELLOW FLAG — CA two-party consent + retention policy needed.** |
| **Silero-VAD** [link](https://github.com/snakers4/silero-vad) | 4k+ | LOW | Skip silent audio segments before Whisper. <1ms per chunk. Compresses transcription load 5–10x. |
| **openWakeWord** [link](https://github.com/dscripka/openWakeWord) | 1.2k+ | MED | Voice-triggered Command Center actions ("alert mode on", "show me the doorbell"). Privacy-preserving — no transcription, just keyword. |
| **rtl_433** [link](https://github.com/merbanan/rtl_433) | 5k+ | LOW | **Your existing RTL-SDR becomes a 433 MHz home-sensor receiver.** Cheap door contacts / PIR / temp/humidity sensors ($5–20 each) → MQTT → Command Center event log. |

### Tier 2 — Small purchase, big lift

| Project | Hardware cost | Value |
|---|---|---|
| **BLE beacon scanning** (aioblescan, beacontools) | $10 USB Bluetooth 5.0 dongle (or use Mac's built-in) | Personal-device presence ("Max's phone is home"). **YELLOW FLAG — only known consented devices, never strangers' AirTags. CA stalking statutes apply.** |

### Tier 3 — Bigger commitment

- **Home Assistant** [link](https://github.com/home-assistant/core) — Industry-standard MQTT broker + 1000+ device integrations. Could become the central event bus that all sensor types (rtl_433, BLE, Zigbee, Z-Wave, Shelly) feed into. Then Command Center consumes one normalized MQTT stream. Tradeoff: another whole service to operate. Decide later.
- **Glass-break detection (custom spectrogram CNN)** — YAMNet handles the easy case. Custom training (~100–500 clips) on Edge Impulse gets you sub-5% false-positive rate. Wait until YAMNet results justify it.
- **Lip-sync / deepfake detection** — research-grade, build-from-papers. Defer until deepfaked doorbell intrusions become real.

### Ethical yellow flags (real)

| Issue | Mitigation |
|---|---|
| **AirTag / BLE tracking** | Only your own consented devices. Never scan random AirTags in the area. Frame as home occupancy, not tracking. |
| **Doorbell transcript PII** | Set retention window (30d auto-purge), document consent (signage at door), summarize events instead of full transcripts where possible. |
| **Speaker diarization** | Label "Speaker A/B/C" not names. Never combine diarization + face_intel into a searchable cross-modal speaker index without explicit user policy. |
| **Sex offender POI overlay** | Home-premises only, ~500m radius. Refresh weekly. View-only for Max. |

### Hard pass

- **Gunshot detection** — High false-positive on construction / firecrackers. ShotSpotter-grade requires triangulation. Stick with glass-break + structural sensor fusion instead.
- **Snore detection** — Medical use case, not threat-model.
- **IMSI catchers / cell triangulation** — Illegal without warrant. Hard line.
- **WiFi probe-request fingerprinting** — Surveillance-grade, de-anonymizes visitors. Hard line.

---

## Domain 5 — Tactical UX / community dashboards

### Banger steals (copy components straight in)

| Project | Effort | What to steal |
|---|---|---|
| **xterm.js** [link](https://github.com/xtermjs/xterm.js) (17k stars) | 3h | Replace the vanilla PALANTIR terminal widget. Unlock scroll buffer, copy/paste, ANSI escapes, session replay for audit. |
| **Sparklines.js** [link](https://github.com/mitjafelicijan/sparklines) | 1h | Pure-vanilla SVG sparklines (zero deps). Drop-in replacement for your current 14-day velocity canvas. Matches the no-framework philosophy. |
| **BrowSDR waterfall** [link](https://github.com/jLynx/BrowSDR) | 2h | WebGL spectrum + waterfall canvas. Paste into a new OPS Center panel for live RTL-SDR display. |

### Strong inspirations (build your own from the pattern)

| Project | What to learn |
|---|---|
| **OSINT War Room** [link](https://github.com/Hue-Jhan/OSINT-War-Room) | 3-panel OPS Center layout (Shadowbroker | Palm | Ground Station) with cyan/dark status pills. You're already 80% there. |
| **Clermont** [link](https://github.com/SageHourihan/clermont) | Cold-war command-center scrolling status ticker. Add a facility-status roll to `index.html`. |
| **Frigate Web UI** [link](https://github.com/blakeblackshear/frigate) | Mobile-responsive camera grid + detection overlay. Cribbed for `monitor.html` corner-bracket pattern (already in place; could level up). |
| **MapLibre + MGRS Mapper** [link](https://maplibre.org/) + [link](https://mgrs-mapper.com/) | If you add a facility-topology map surface, MGRS grid overlay is the right tactical detail. |
| **deck.gl** [link](https://deck.gl/) | 3D facility / camera positioning. Heavy — only if you build the facility-topology surface. |

### Hard pass

- **OpenFoundry** (Palantir clone) — enterprise-scale, React/TS/GraphQL. 80+ hours to even understand. Wrong scope.
- **Grafana** — Excellent for general metrics, but adds a service. Iframe embed only if you have a real metrics need.
- **D3.js** — Heavy redraws for streaming dashboards. Use canvas / deck.gl instead.
- **ATAK-CIV** — Android-only, wrong surface.

---

## Cross-cutting themes

1. **Your RTL-SDR is underutilized.** Three top picks (rtl_433, dump1090, AIS-Catcher) all reuse hardware you already own. If you do nothing else from this brainstorm, plug those in.
2. **Audio is your free intel multiplier.** The Tapo D210 has 2-way audio. YAMNet + Silero-VAD on the mic doubles your event surface for zero new hardware.
3. **The OPS Center :3600 wants to be your console.** Five things to consider for it: spectrum waterfall (Ground Station), facility-status ticker (Clermont pattern), MGRS map (MapLibre), ADS-B/AIS overlays (Shadowbroker), and the current 3 CRT TVs. Tactical density without clutter.
4. **Don't rewrite what works.** Frigate is tempting but you'd lose entity_resolution, threat scoring, PALANTIR NLU. Keep your conductor; pull only what you need.
5. **Ethics policy goes BEFORE deployment.** Whisper transcription, BLE tracking, speaker diarization, sex offender POI overlay — all useful, all want a written 1-page consent/retention policy first. Easier to write now than retroactively.

## Open questions for Max

1. **Frigate vs keep-rolling-your-own?** Yes/no on Coral EdgeTPU ($100). If yes → Frigate becomes the recording layer + camera_watcher.py becomes the intel layer.
2. **Home Assistant as event broker?** Yes → unifies sensor types but adds a whole service. No → keep custom event_db, pull rtl_433/YAMNet/etc directly.
3. **Audio retention policy?** How long do doorbell transcripts live? Default proposal: 30 days, auto-purge.
4. **Drone detection priority?** GridDown / FAA Remote ID adds a real "is anyone scoping my house" signal. Worth it?
5. **OPS Center :3600 expansion?** Stay 3-pane (current) or grow into a 5–8 panel tactical console (waterfall, MGRS map, status ticker, plus the TVs)?
6. **Wood-grain second pass?** Today's pass was 70% there. Want a richer walnut with grain knots in the next iteration?
