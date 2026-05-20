---
date: 2026-05-19 → 2026-05-20
shift: 9pm Tue → 6am Wed (autonomous overnight while Max at facility)
scope: Phase D (Palm Command Phase 0/1 addons + Sentinel hardening v2)
plan_doc: docs/SIDE-ADDONS-PLAN-2026-05-20.md
---

# Overnight HANDOFF — read this first at 6am

## TL;DR (one-line per change)

_(populated as each task ships)_

## Open decisions for Max (eyes-on items)

_(populated when a task hits a fork I can't take alone)_

## Things I deliberately did NOT touch

- Wood-grain bezel v2 (Phase 0.4) — aesthetic call, your eyes
- Anything on the facility side (CannaMax, FT, Pi, etc.) — out of scope for tonight
- POI / face / retention values in any policy doc — drafted as DRAFT only
- HARK Map node positions (just unblocked the JSON, didn't re-layout)
- Shadowbroker — see "noticed but didn't touch" below
- **Doorbell cameras.yaml poll/cooldown profile** (PLAN Phase 0.1) — deferred because I couldn't autonomously verify USB-C is wired. Obs from earlier today said `Power source: BATTERY`. When you confirm USB-C is plugged in, the revert is trivial.

## ⚠️ Noticed and need your eyes

### Two diverged copies of Palm Command source on disk

- `~/palm-command/` — where the PM2 dashboard process runs `serve_dashboard.py` from
- `~/Documents/GitHub/command-center/` — git remote (TheMaxLong/command-center) AND where the running Docker stack `palm-vision-watcher` was launched from

CLAUDE.md still asserts "All source lives in `~/palm-command/` and is synced to this repo" — that's no longer true. Files differ (docker-compose.yml, serve_dashboard.py both diverged). Tonight I made edits to **command-center** for git-trackable / Docker-affecting work (docker-compose mount, start.sh, scripts/, docs/), and to **palm-command** for the live-PM2-affecting work (start.sh — same edit synced to both). HANDOFF doc lives in `command-center/docs/`.

**Recommendation when you have 5 min:** decide which is canonical and symlink the other (or reconcile by hand). They'll keep drifting.

## Noticed but didn't touch

- **`shadowbroker-backend` is in a restart loop** (exit 137, OOM-killed every ~30s). Container belongs to the Shadowbroker project, not Palm Command. Flagged here in case it's news to you; happy to chase tomorrow.

---

## Baseline at 9:30pm (before any changes)

| Surface | State |
|---|---|
| `serve_dashboard.py` PM2 process | online, 5d uptime, pid 39683 |
| `palm-command-streams` (go2rtc) | up 3h, ports 1984/8554-5/8555 |
| `palm-vision-watcher` | up 6h, port 8181 |
| Dashboard `:8888/` | HTTP 200, 186KB |
| `/api/events` via proxy | HTTP 200, latest event #39 doorbell |
| Doorbell night vision | `wtl_night_vision` (last frame 103.5/255 brightness) |
| Sentinel pi-health (Pixel 6) | `root=up flux=up ft=up` |
| Sentinel hark-map watcher | clean snapshot 20:23:28 after JSON fix |
| Sentinel cannamax watcher | `cannamax-endpoint-unreachable` since 18:45 (pre-existing, in queue) |

## Per-task log

### ✅ Task #1 — handoff doc + baseline (9:30pm)

- HANDOFF doc created at `docs/HANDOFF-2026-05-19-overnight.md`.
- Stack verified healthy (table above).
- One out-of-scope anomaly noted (shadowbroker-backend OOM loop).
- Verification: `curl /api/events` via dashboard proxy returns latest doorbell event JSON.
- Rollback: N/A (no changes).

### ✅ Task #2 — doorbell-brightness watcher (Mac-side launchd) (9:37pm)

- New script: `~/bin/doorbell-brightness-watcher.sh`
- New plist: `~/Library/LaunchAgents/com.max.doorbell-brightness-watcher.plist` (loaded, PID 95998)
- Polls `/api/snap/doorbell` (latest captured event snapshot, NOT a live frame — battery doorbells sleep)
- Computes avg pixel brightness via ffmpeg. Threshold: avg <2.0% of 255 for 3 consecutive checks → ntfy alert on `hark-phones-b6d6f70e9913` (your phone is already subscribed).
- Skips check entirely if last motion event is >12h old (no fresh signal to evaluate).
- Recovery sends a separate ntfy when brightness comes back.
- State at `~/.local/state/doorbell-brightness.json`. Log at `~/.local/state/doorbell-brightness.log`.
- **Verification:** alert path test (3 black-frame strikes → ALERT FIRED on strike 3) and recovery path test (seeded ALERTED state + bright frame → RECOVERY) both passed end-to-end with sandbox ntfy topic.
- Current state at boot: `33.06%` brightness, `OK`, 0 strikes.
- **Rollback:** `launchctl unload ~/Library/LaunchAgents/com.max.doorbell-brightness-watcher.plist && rm ~/Library/LaunchAgents/com.max.doorbell-brightness-watcher.plist ~/bin/doorbell-brightness-watcher.sh ~/.local/state/doorbell-brightness.*`

### ✅ Task #3 — Pixel 6 sshd belt-and-suspenders

- **On-device** (Termux cron, every 1 min): `~/sshd-watchdog.sh` — `pgrep sshd || sshd`. Auto-respawns. Deployed from `phone-infra/sshd-watchdog.sh` in HARK-System-Map repo.
- **Off-device** (Mac launchd, every 5 min): `~/bin/pixel6-sshd-watchdog.sh` + `com.max.pixel6-sshd-watchdog.plist` (PID 98780). Probes `100.72.211.71:8022`. 2-strike threshold → ntfy alert on `hark-phones-b6d6f70e9913` with literal recovery instruction.
- Suppresses alerts if Mac's Tailscale is down (avoids false alarms on coffee-shop wifi).
- **Verification:** alert path tested with unreachable port (127.0.0.1:1) → fired ALERT FIRED after 2 strikes. Healthy state shows `{"strikes":0,"state":"OK"}`.
- **Rollback:** unload Mac plist + remove watchdog script; `crontab -e` on Pixel 6 to drop `* * * * * ~/sshd-watchdog.sh` line.

### ✅ Task #4 — Sentinel: auto-park cannamax watcher

- **Problem:** `cannamax-endpoint-unreachable` digest noise since 18:45 because CannaMax isn't deployed anywhere reachable. URLs in script were placeholders ("Endpoint path TBD").
- **Fix:** `phone-infra/sentinel/watch-cannamax.sh` (in HARK-System-Map repo) now reads `~/sentinel/cannamax.url` (on Pixel 6). If file missing/empty → silent exit. No URL = no noise.
- **To re-enable when CannaMax deploys:** `ssh pixel6 'echo "https://your-url/api/change-events?since=15m" > ~/sentinel/cannamax.url'`
- Watcher then polls the configured URL and resumes drift alerts.
- **Verification:** ran on Pixel 6 with no URL file → exit 0, no digest entry.

### ⏸️ Task #5 — Doorbell battery profile (PLAN 0.1) — DEFERRED

Need your confirmation that the doorbell is USB-C wired. Pytapo probe at 22:03 returned `Power source: BATTERY`. Cranking `poll_interval` to 3 on battery will drain it.
**When you confirm:** edit `cameras.yaml` → `poll_interval: 3`, `cooldown: 300`, `capture_duration: 8`, then `docker compose up -d --force-recreate vision-watcher`.

### ✅ Task #6 — Doorbell night vision self-lock (Phase 0.2)

- **New:** `scripts/ensure-night-vision.py` in command-center. Idempotent: reads current `night_vision_mode` via pytapo. If `wtl_night_vision`, exits silently. If drifted, re-applies and logs. If camera asleep (battery doorbells sleep!), exits 0 silently.
- **Mount:** docker-compose.yml now binds `./scripts:/app/scripts:ro` so the container has access to pytapo for this script.
- **start.sh:** backgrounded call to `docker exec palm-vision-watcher python3 /app/scripts/ensure-night-vision.py` (4s after compose up). Synced to both `~/palm-command/start.sh` AND `~/Documents/GitHub/command-center/start.sh` due to the diverged-copies issue above.
- **Launchd:** hourly `com.max.palm-night-vision-lock` (PID 10978) runs the same docker exec.
- Log at `~/.local/state/palm-night-vision.log` — quiet on the happy path; only writes on drift detection / re-application / errors.
- **Verification:** container recreate succeeded with new mount; ensure-night-vision.py executed from inside container (asleep-camera branch hit, exit 0 silently). Dashboard `/api/events` returned 200 immediately after.
- **Rollback:** unload `com.max.palm-night-vision-lock.plist`; remove scripts mount from docker-compose.yml; restore start.sh from git. The script itself can stay — harmless if unused.

### ⏸️ Task #8 — Sparklines.js swap (PLAN 1.4) — NOT APPLICABLE

PLAN doc said "replace canvas sparkline in 5-WEEK TRENDS tab." Verified there's no `<canvas>` left in `dashboard/index.html` — the trends tab is already pure HTML/CSS div grid + bars. Premise outdated; nothing to swap. Could add a daily-velocity sparkline as a NEW feature later — flagging it for your call.

### ✅ Task #9 — YOLOv10 bench (PLAN 1.1) — KEEP yolov8s.pt

- **New:** `scripts/bench-yolo.py` — runs 4 model variants (v8n, v8s, v10n, v10s) on 15 sampled archived doorbell snaps. Conf 0.10 to capture borderline detections.
- **Findings (15 frames, conf≥0.10, run inside container):**

  | Model | Median infer | Total dets | Person dets | Person conf |
  |---|---|---|---|---|
  | yolov8n.pt | 88ms | 6 | 0 | — |
  | **yolov8s.pt (current)** | **228ms** | **7** | **3** | **0.155** |
  | yolov10n.pt | 95ms | 3 | 1 | 0.155 |
  | yolov10s.pt | 238ms | 3 | 0 | — |

- **Recommendation: KEEP yolov8s.pt.** YOLOv10 was faster but detected ~50% fewer things on this dataset (low-light night frames). PLAN's "10–15% faster + better accuracy" did not hold here. No swap. Bench script lives in `scripts/bench-yolo.py` — re-run after daylight motion events to confirm with brighter frames.
- **Rollback:** N/A (no code change to ai_engine.py).

### ✅ Task #10 — Cross-cut: SENTINEL tile in Palm Command

Three pieces, all reversible:

1. **Mac-side sync** (`~/bin/sentinel-digest-sync.sh` + `com.max.sentinel-digest-sync.plist` launchd, every 5 min): pulls last 50 lines of `~/sentinel/digest.log` from Pixel 6 over Tailscale SSH → writes to `~/.local/state/sentinel-digest.tail`. Survives Pixel 6 going to sleep — leaves last known good content in place.
2. **New endpoint** `GET /api/sentinel-digest` (in `serve_dashboard.py`): parses the tail into JSON `{last_synced, entries:[{ts,level,watcher,message}]}`. Newest-first. Local-only handler — does NOT proxy to vision-watcher.
3. **New AI Intel tab** `SENTINEL` in `dashboard/index.html`: color-coded entries (info=cyan, warn=orange, critical=red), sync age in header colored by staleness (>15min red, >7min orange).

- **Verification:** Playwright end-to-end — opened dashboard, called `openAIPanel('sentinel')`, asserted header present, sync status shown, pi-health entry present, hark-map entry present, no UNAVAILABLE error. All 5 checks PASS. Screenshot at `/tmp/sentinel-tab.png`.
- **Both source dirs in sync** (palm-command + command-center).
- **Rollback:** unload sentinel-digest-sync plist; revert the serve_dashboard.py + index.html edits via `git checkout`; remove the watcher script.
