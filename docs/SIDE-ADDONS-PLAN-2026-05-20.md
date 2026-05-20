---
date: 2026-05-20
status: phased plan — for Max's morning read
scope: HOME ONLY (Palm Springs residence). Tap-in from facility via Tailscale.
companion: SIDE-ADDONS-BRAINSTORM-2026-05-20.md (full domain research)
---

# Palm Command — Side Add-Ons Plan

Phased rollout. Each item has effort estimate, value, dependency, and decision-needed flag. The phases are sequenced so each one ships independently — you can stop at any phase and the system is still cleaner than where you started.

**Sequencing rules:**
- Phase 0 = pure config / settings changes, no new dependencies
- Phase 1 = drop-in replacements, no architectural change
- Phase 2 = new feature surfaces (audio, SDR, OSINT feeds)
- Phase 3 = deeper integration (Frigate / Home Assistant — architectural)
- Phase 4 = speculative / future-flag

**Effort scale:** ⏱ ≤ 1h · ⏱⏱ 1–4h · ⏱⏱⏱ 4–16h · ⏱⏱⏱⏱ 16h+

---

## Phase 0 — Free wins, do tomorrow (≤ 2 hours total)

These are all settings changes or one-file additions. No new dependencies, no architectural shift.

| # | Item | Effort | Value | Decision |
|---|---|---|---|---|
| 0.1 | **Revert doorbell battery-conservation profile** in `cameras.yaml` now that USB-C is wired: `poll_interval: 3` (was 15), `cooldown: 300` (was 600), `capture_duration: 8` (was 6) | ⏱ | Faster motion response on doorbell, more clips per event | NONE — just do it |
| 0.2 | **Lock the doorbell `night_vision_mode: wtl_night_vision`** via a startup script that re-applies the setting (in case firmware resets it). Add to `start.sh`. | ⏱ | Prevents regression to the black-frame `dbl_night_vision` bug we fixed today | NONE |
| 0.3 | **Document audio/recording consent policy** — 1-page text file at `docs/POLICY.md`. Retention windows (clips 14d, transcripts 30d, faces indefinite, gait indefinite). Door signage requirements. | ⏱ | Legal cover before Phase 2 audio work. CA two-party-consent territory. | Max writes / approves wording |
| 0.4 | **Wood-grain CRT bezel v2** — add knot/grain variation, deeper walnut, brushed-brass label inlays | ⏱⏱ | Visual polish; Max's design sensibility | Max signs off on v2 vs current |

**Total: ~3 hours of work. Highest "shipped today" leverage.**

---

## Phase 1 — Drop-in upgrades (1–2 sessions, ~10 hours)

Drop-in replacements that don't change behavior for users but level up internals. No new architecture.

| # | Item | Effort | Value | Risk |
|---|---|---|---|---|
| 1.1 | **YOLOv10 swap** in `ai_engine.py`: one-line `YOLO('yolov10n.pt')`. Bench against existing. | ⏱⏱ | 10–15% faster + better small-object accuracy (faces, plates) | Low. Trivial rollback. |
| 1.2 | **Supervision (Roboflow)** for annotation + ByteTrack glue. Replaces ~150 lines of `ai_engine.py` boilerplate. | ⏱⏱⏱ | Cleaner internals, easier future tracker swaps | Low. Production library, 39k stars. |
| 1.3 | **InsightFace (ArcFace 512-dim)** replaces 80-dim histogram in `face_intel.py`. Re-enroll existing POI photos. | ⏱⏱⏱ | ~20% face-match accuracy improvement | Low–med. Re-enrollment of existing POI faces required. Check model license per ship target. |
| 1.4 | **Sparklines.js** (pure vanilla SVG) replaces canvas sparkline in `index.html` 5-WEEK TRENDS tab | ⏱ | Cleaner UX, smaller code, matches no-framework philosophy | Trivial. |
| 1.5 | **xterm.js** for the PALANTIR terminal in `index.html` | ⏱⏱⏱ | Scroll buffer, copy/paste, ANSI escapes, session-replay capability | Low. 17k-star library, MIT. |
| 1.6 | **WebSocket subscription motion events from Tapo front_cam** via `python-onvif-zeep`. Replaces "TCP port open = motion" inference for the C-series. | ⏱⏱⏱ | Real motion events (10x more reliable than polling), lower CPU | Med. Tapo D210 doorbell does NOT support this — front cam only. |

**Total: ~13 hours. After this phase: same UX, much cleaner + more accurate internals.**

---

## Phase 2 — New feature surfaces (2–4 sessions, ~20 hours)

New capabilities. Each is independent — you can ship any subset.

### 2A — Audio intel layer (your highest-leverage Phase 2 work)

| # | Item | Effort | Value | Notes |
|---|---|---|---|---|
| 2.1 | **YAMNet** on Tapo D210 mic. 521 audio classes — start with dog-bark / glass-break / siren / smoke-alarm event tagging | ⏱⏱⏱ | Doubles event surface for $0 new hardware | New SQLite table: `audio_events` |
| 2.2 | **Silero-VAD** front-end (before any transcription) to skip silent segments | ⏱ | Compresses load 5–10x | Pure stdlib chain into YAMNet/Whisper |
| 2.3 | **faster-whisper** transcription of doorbell conversations, tagged to clips | ⏱⏱⏱ | Audit trail of porch interactions | **REQUIRES POLICY.md from Phase 0.3.** CA two-party consent. Default: 30d auto-purge. |
| 2.4 | **openWakeWord** voice-trigger for Palm Command actions ("alert mode on", "show doorbell") via Mac built-in mic | ⏱⏱⏱ | Hands-free dashboard control | Optional. Test in single-user mode first. |

### 2B — SDR / RF intel layer (uses your existing RTL-SDR — Ground Station hardware)

| # | Item | Effort | Value | Notes |
|---|---|---|---|---|
| 2.5 | **rtl_433** receiver + MQTT broker, ingest 433 MHz home sensors | ⏱⏱⏱ | Cheap door/PIR/temp sensors anywhere ($5–20 each). Massive coverage expansion for $0 core infra | Requires installing a small MQTT broker (mosquitto, tiny) |
| 2.6 | **dump1090 / readsb + tar1090** ADS-B receiver. Aircraft tracking near your home. | ⏱⏱⏱ | Shadowbroker integration. PSP arrivals + military traffic context | Likely already partly running in Ground Station — check first |
| 2.7 | **AIS-Catcher** maritime tracker (dual-band) | ⏱⏱ | Salton Sea + dust-storm context | Same hardware, same Ground Station rig |
| 2.8 | **GridDown drone detection** via FAA Remote ID | ⏱⏱⏱ | Real "is someone droning my house" signal | Decision needed: priority? |
| 2.9 | **gpredict + CelesTrak TLE** auto-update for satellite passes | ⏱⏱ | Pairs with satdump for NOAA, ISS predictions | Read-only, polls every 6h |

### 2C — Public-data overlays

| # | Item | Effort | Value | Notes |
|---|---|---|---|---|
| 2.10 | **Blitzortung lightning MQTT** subscription, 50km radius | ⏱⏱ | Desert monsoon early warning | Lightweight; just adds a feed to `intel_feeds.py` |
| 2.11 | **PurpleAir API** AQI for surrounding neighborhood mesh | ⏱⏱ | Wildfire-smoke + dust-storm context | Same `intel_feeds.py` pattern |
| 2.12 | **California Open Data — Power Outage Incidents API** | ⏱⏱ | Real-time SCE outages geofenced to your address | Pairs with NWS heat alerts |
| 2.13 | **NSOPW (sex offender registry)** scrape, 500m POI overlay | ⏱⏱⏱ | Home-premises defensive awareness | Weekly refresh; HOME ONLY scope |
| 2.14 | **AbuseIPDB** IP reputation lookup on inbound network logs | ⏱⏱ | If you ever expose anything beyond Tailscale | Free tier 1k/day |

### 2D — Tactical UX additions

| # | Item | Effort | Value |
|---|---|---|---|
| 2.15 | **BrowSDR waterfall** as a 4th OPS Center panel showing live RTL-SDR spectrum | ⏱⏱ | Visual SDR display alongside the 3 CRT TVs |
| 2.16 | **Clermont-style status ticker** on `index.html` footer (scrolling facility-status roll — but for home: doorbell state, AQI, weather alert level, lightning distance) | ⏱⏱⏱ | At-a-glance situational awareness |
| 2.17 | **OPS Center wood-grain v3 if v2 lands** — knots, deeper walnut, brushed-brass label inlays | ⏱⏱ | Aesthetic depth |

**Total Phase 2: ~50 hours if every item shipped. Realistic subset: 2.1 + 2.2 + 2.5 + 2.6 + 2.10 + 2.11 + 2.15 = ~20 hours and a transformed system.**

---

## Phase 3 — Architectural (decisions before code, ~40 hours)

These change how the system is shaped. Don't start without an explicit yes.

| # | Item | Effort | Value | Decision needed |
|---|---|---|---|---|
| 3.1 | **Frigate NVR as 24/7 recording layer** behind `camera_watcher.py`. Frigate handles continuous record + index; your conductor still does entity resolution / threat scoring / NLU. Buy Coral EdgeTPU (~$100) for CPU-friendly inference on the Air bridge. | ⏱⏱⏱⏱ | 24/7 recorded footage indexed by object class. Disk space grows fast (~50–100GB/cam/month). | Yes/no on Coral purchase. Yes/no on disk growth. |
| 3.2 | **Home Assistant as MQTT event broker** for all sensor types (rtl_433, BLE, Zigbee, Z-Wave, Shelly). Palm Command consumes one normalized stream. | ⏱⏱⏱⏱ | Cleaner integration story. Tradeoff: another whole service to operate. | Yes/no on adding HA. |
| 3.3 | **Scrypted as camera abstraction** in front of go2rtc — adds HomeKit Secure Video bridge if you want native iOS viewing | ⏱⏱⏱⏱ | Apple Home integration. Tradeoff: replaces the camera layer that already works. | Yes/no on HKSV (requires HomePod/Apple TV in the home). |
| 3.4 | **MapLibre + MGRS overlay** for a facility-topology surface (where each camera is positioned in the house, sensor coverage zones, dead zones) | ⏱⏱⏱⏱ | Spatial awareness of camera coverage | Yes/no on building a "topology" surface at all |

---

## Phase 4 — Speculative / future-flag (don't start without explicit ask)

| # | Item | When it's worth it |
|---|---|---|
| 4.1 | **YOLO-World open-vocab** detection | When "detect: person carrying weapon" without retraining becomes a real need |
| 4.2 | **FastReID** 2048-dim person re-ID | When you have 3+ cameras and need cross-room tracking |
| 4.3 | **MMPose RTMPose-tiny** | If gait accuracy plateaus with current YOLOv8n-pose |
| 4.4 | **SlowFast action recognition** | "Running / fighting / kneeling" detection. Requires temporal buffer rearchitecture. |
| 4.5 | **MobileSAM** background segmentation | If face/pose inference is being polluted by busy backgrounds |
| 4.6 | **Lip-sync / deepfake detection** | When deepfaked doorbell intrusion becomes a real threat model |
| 4.7 | **Custom glass-break spectrogram CNN** | If YAMNet's glass class produces too many false positives |
| 4.8 | **DJI M30T drone integration** (already in `drone_ops.py` as stub) | After Max actually buys the drone |
| 4.9 | **deck.gl 3D facility topology** | After 3.4 ships and 2D feels insufficient |
| 4.10 | **MMTracking / OC-SORT swap** | If ByteTrack ever stops working for the scene |

---

## Tomorrow morning's recommended order

If you have 2 hours:
- Phase 0.1 + 0.2 + 0.3 — settings + policy
- Phase 1.1 (YOLOv10 swap)
- Air-as-bridge install (separate task)

If you have a weekend:
- Add Phase 1.4 (sparklines), 1.5 (xterm.js), 2.5 (rtl_433), 2.10 (Blitzortung), 2.11 (PurpleAir), 2.15 (BrowSDR waterfall in OPS Center)

If you have a focused week:
- Everything in Phase 1 + Phase 2A (audio layer) + 2B (SDR layer)
- That's the realistic version of a "transformed Palm Command"

---

## Open decisions blocking later phases

1. **Coral EdgeTPU purchase ($100)?** Gates Phase 3.1 Frigate.
2. **Home Assistant — yes/no?** Gates 3.2 + sensor sprawl architecture.
3. **HomeKit Secure Video — care?** (Need HomePod/Apple TV) Gates 3.3.
4. **Topology surface — care?** Gates 3.4 + 4.9.
5. **Drone detection priority — Phase 2 or 4?** Affects 2.8 ordering.
6. **DJI M30T — when?** Unblocks 4.8.

Answer these in any order; each unlocks downstream phases.

---

## Hard passes from the brainstorm (documented so we don't re-research)

- Tapo D210 button-press API — does not exist
- MotionEye — abandoned
- OpenALPR — AGPL legal trap
- Shodan — wrong tool for hyperlocal
- Citizen API replacement scrape — TOS / legal
- Broadcastify — not OSS, unreliable
- Gunshot detection — false-positive rate too high
- WiFi probe-request fingerprinting — surveillance-grade, ethical line
- IMSI catcher / cell triangulation — illegal without warrant
- OpenFoundry — enterprise-scale Palantir clone, wrong scope
- ATAK-CIV — Android only, wrong surface
- mediasoup — overkill for <20 viewers
