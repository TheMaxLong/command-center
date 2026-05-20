---
status: DRAFT — needs Max's review + approval before any audio feature ships
scope: HOME ONLY (Palm Springs residence). All cameras + sensors + audio at home.
last_revised: 2026-05-19 overnight (Claude, autonomous)
plan_doc: docs/SIDE-ADDONS-PLAN-2026-05-20.md
---

# Command Center — Data & Consent Policy (DRAFT)

This document is the operational consent / retention policy for the Command Center home surveillance stack. It is **prerequisite** to shipping any audio capture / transcription feature (PLAN Phase 2.3 onward).

> **This is a working draft, not legal advice.** Items marked **(legal: review)** need a quick read by an attorney before going live, especially anything touching California two-party-consent (Penal Code § 632), CCPA, or the California stalking statute. Max should not treat this doc as a substitute for that review.

## 1. What we capture

| Source | Type | Purpose | Currently live? |
|---|---|---|---|
| Tapo D210 doorbell | Video + still frames | Motion detection, scene analysis, person re-ID, face recognition (FBI watchlist), threat scoring | ✅ Yes |
| Tapo D210 doorbell | Audio (2-way) | YAMNet event classification (bark / glass-break / sirens) | ❌ Not yet (PLAN 2.1) |
| Tapo D210 doorbell | Audio (speech) | faster-whisper transcription of porch conversations | ❌ Not yet (PLAN 2.3 — blocked on this policy) |
| Front camera (Tapo C-series) | Video | Same as doorbell | ✅ Yes |
| RTL-SDR (Ground Station) | RF (433 MHz / ADS-B / AIS) | Sensor mesh, aircraft, maritime | ❌ Not yet (PLAN 2.5/2.6) |
| Mac built-in mic | Audio (wake word) | openWakeWord trigger ("alert mode on") | ❌ Not yet (PLAN 2.4) |
| BLE scan (Mac built-in) | RF | Personal device occupancy (Max's phone) | ❌ Not yet |

## 2. Retention windows

| Data | Retention | Auto-purge | Notes |
|---|---|---|---|
| Video clips (motion events) | 14 days | Yes — `ARCHIVE_RETENTION_DAYS` env | Configurable per camera |
| Still snapshots | 14 days | Yes | Same env var |
| SD-card backfill clips | 30 days | Manual `docker compose run backfill --prune` | Larger window because backfill is bursty |
| Audio event tags (YAMNet output, no waveform) | 90 days | Manual | Just labels + timestamps, no recording |
| Audio transcripts (Whisper) | **30 days** | Yes — to-be-built | **(legal: review)** CA two-party consent. Summary may persist; full transcript purges. |
| Face embeddings | Indefinite | Manual delete only | Vector only, not the image. Image follows clip retention. |
| Face images (when matched to a profile) | Indefinite | Manual delete only | Linked to person profile in `event_db`. |
| Gait embeddings | Indefinite | Manual delete only | 18-dim skeleton vector. |
| License plate reads | 90 days | Yes — to-be-built | Plate text + timestamp + camera_id. |
| Detection logs / events table | 1 year | Manual | Structured event records; small footprint. |

## 3. Consent posture (California)

California is a **two-party consent** state for recording of confidential communications (Penal Code § 632). This affects:

- **Audio capture at the door** — anyone speaking on the porch is recorded.
- **Audio capture in any room** — same.
- **License plate recording** — generally permitted from a private residence (one-party not required since plate is public); commercial / large-scale collection has separate rules. **(legal: review)**

### Required before turning on audio:

1. **Door signage** — posted at the door, visible before someone presses the doorbell. Suggested text:
   > "AUDIO + VIDEO RECORDING IN PROGRESS. Press the doorbell to communicate."
   This converts the encounter from "confidential communication" to "knowing recording."

2. **No covert mics inside the home** beyond what the doorbell + cameras already capture.

3. **Default-off for transcription** — YAMNet event classification can run without consent issues (it's not "recording" speech, it's classifying sound categories). Whisper transcription is the line: do not enable for production until signage is up and this policy is reviewed.

4. **Speaker diarization** — label speakers as "Speaker A / B / C" only. Do NOT combine diarization with `face_intel` to build a cross-modal speaker-identity index without an additional explicit policy revision. **(legal: review)**

### What Max should never enable without an additional policy revision:

- **Random AirTag / BLE scanning** beyond Max's own consented devices (CA stalking statutes — Penal Code § 646.9).
- **WiFi probe-request fingerprinting** of unknown devices (surveillance-grade; de-anonymizes visitors).
- **IMSI catchers / cell-tower triangulation** (illegal without warrant — hard line).
- **Lip-reading / silent speech detection from video** (no clean legal precedent; treat as transcription).

## 4. Access control

| Audience | Access |
|---|---|
| Max | Full read/write |
| Anyone else | None — Tailscale-only network access. No public endpoints. |

- `PALM_API_TOKEN` env var gates external requests to `:8181`. Localhost always bypasses.
- Mobile views (`/mobile`, `/field`) accessible only over Tailscale.
- No third-party telemetry. No vendor cloud uploads (Tapo cloud explicitly NOT enabled).

## 5. Watchlists & POIs (sensitive)

- **FBI Wanted face database** — `face_intel.py` matches against LA/SD/LV field offices. View-only; matches log to alerts; no upstream reporting.
- **Sex offender registry overlay** (NSOPW) — PLAN 2.13, not yet built. When built: home-premises radius (500m), weekly refresh, view-only for Max, no UI exposure to anyone else.
- **License plate watchlist** — local list only. No external sharing.

## 6. Logs and audit trail

- All face matches, plate hits, threat-tier escalations, and notifier dispatches log to `events` and `alerts` tables in `event_db`.
- The PALANTIR terminal queries are logged to `manual_scans` (NLU intent + query string).
- Logs are local to the Mac. Backups (if any) stay local or on a Tailscale-only host.

## 7. Update process

- Material changes to retention or capture surface require a new dated section here + a commit.
- Audio feature rollouts (PLAN 2.x) require this doc to be at status `APPROVED` (not `DRAFT`).
- Annual review (or after any incident).

## 8. Things Max owes this doc

- Read it.
- Approve / edit retention numbers (best guesses now).
- Sign off on door signage wording.
- One legal pass before flipping the audio kill switch.
