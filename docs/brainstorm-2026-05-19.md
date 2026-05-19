# Brainstorm: Spec Ops Integration + MoCap Pipeline — 2026-05-19

## Hardware Moment

You now have:
- **Seagate Portable Drive** — 926GB total, 572GB free. Dedicated offline storage.
- **Wired Doorbell** — Tapo with RTSP. Real motion events at the house perimeter.
- **FreeMoCap Installer** — Ready to run on demand. Turns video → skeleton BVH + movement data.
- **OPS Center** — New 3-pane unified dashboard. Shadowbroker + Palm Command + Ground Station at :3600.

The pieces are here. Now think invasive.

---

## Invasive Thinking: MoCap as Signal Intelligence

### Thesis
Motion capture is not just animation. It's **identity without face.** Skeleton data (17 keypoints per frame) is:
- **Unique per person** — gait signature, shoulder width, arm length, movement cadence
- **Invariant to clothing** — body shape doesn't change
- **Invariant to weather/lighting** — works in darkness
- **Persistent** — you can re-identify people across days or weeks
- **Forensic** — timestamp + skeleton chain = movement timeline

### Scenario 1: Doorbell Event → Skeleton Timeline

```
Doorbell motion trigger (Tapo)
  ↓
Auto-clip 8 seconds video → Seagate:/mocap-in/
  ↓
FreeMoCap async batch (headless)
  ↓
Output: skeleton.bvh + preview.mp4 → Seagate:/mocap-out/<event-id>/
  ↓
Dashboard polls for completion
  ↓
Render skeleton on preview tile (Babylon.js? Cesium?)
  ↓
Extract gait vector (18-dim from 17 keypoints)
  ↓
Cross-reference against known profiles (Palm Command gait_engine.py)
  ↓
If match: "PERSON_X approached door at 19:42, dwell 3.2s, left west"
```

**Product value:** You don't need cameras everywhere. One doorbell + skeleton matching = perimeter awareness.

---

### Scenario 2: Field Scan → Instant Gait ID

Your Field Scan app (/field) already does face + LPR from phone camera. Add skeleton:

```
Field Scan: tap camera → video frame
  ↓
Send to FreeMoCap pipeline OR stream to Palm Command
  ↓
Extract skeleton on receipt
  ↓
Query against 30-day skeleton archive
  ↓
Show: "76% match to PERSON_X (gait), 45% face match"
```

This solves the "who is that person?" problem when face fails (cap, sunglasses, distance, angle).

---

### Scenario 3: Cross-Pane Intel Correlation

The new OPS Center is a **tactical fusion surface.** Use it:

```
Palm Command pane (left):
  → Detects unknown person at NW corner camera
  → Pulls skeleton BVH for that moment

Shadow Broker pane (center):
  → Queries AIS + gait database
  → Flags if trajectory matches known threat

Ground Station pane (right):
  → Displays gait similarity chart
  → Links to 30-day timeline of this skeleton
```

**One click:** From detection → geospatial context → threat scoring → action.

---

### Scenario 4: Archive as 30-Day Skeleton Library

The Seagate drive becomes your **skeleton library:**

```
Seagate:/command-center/
├── footage/           ← rolling clips (overwritten every 30d)
├── mocap-in/          ← queued videos awaiting processing
├── mocap-out/         ← processed BVH + previews
├── faces/             ← face crops from detections
├── gait-vectors/      ← 18-dim CSV: timestamp | skeleton_id | vector
├── events/            ← indexed motion events
└── archive/           ← cold storage (compress after 7d)
```

**Query:** "Show me all skeletons with arm span 71-74cm, who visited in the last 21 days."
Output: gait vector search → 3 candidates → timeline overlay.

---

### Scenario 5: Doorbell + MoCap + LPR Fusion

The wired doorbell is the kill zone for identity:

```
Doorbell motion
  ↓
Parallel: LPR (license plate), skeleton (person), audio (optional)
  ↓
Trigger: {vehicle_plate, person_gait, entry_time, dwell, exit_vector}
  ↓
Cross-reference: Is this a known neighbor? Known delivery pattern?
  ↓
If anomalous: RED alert → SMS + slack + siren
  ↓
If known: soft log "regular_mail_carrier 19:47–19:51 Thu"
```

**Outcome:** You'll know if someone who doesn't match a known pattern is at your door.

---

## Feature Shortlist (Priority Order)

### Immediate (This week)
1. **MoCap Pipeline Scaffold**
   - Drag-drop zone on Palm Command dashboard
   - Video → Seagate:/mocap-in/ → FreeMoCap async → BVH/preview output
   - Inline preview tile
   - Test with real doorbell clip

2. **OPS Center Status Widget**
   - Currently shows LIVE/OFFLINE per pane
   - Add: frame rate, latency, last-updated timestamp per pane
   - Optional: SLA pills (GREEN OK / YELLOW WARN / RED ERR)

3. **Skeleton Preview Renderer**
   - BVH viewer in dashboard (Babylon.js skeleton mesh)
   - Play/pause, timeline scrub
   - Overlay gait vector as heatmap on joints

### Short-term (2-3 weeks)
4. **30-Day Skeleton Archive Index**
   - SQLite table: {timestamp, skeleton_id, gait_vector, entry_video, person_id}
   - Gait cosine search (≥0.88 = match)
   - Quick query from dashboard

5. **Gait Field Scan Integration**
   - Field app: extract skeleton on phone from video frame (edge inference? or send to backend?)
   - Show similarity score vs archive
   - Link to full timeline for that skeleton ID

6. **Doorbell Auto-Clip to MoCap**
   - Tapo motion event → auto-save to mocap-in/ (timestamp-named)
   - Auto-trigger FreeMoCap batch
   - Webhook callback to dashboard on completion

### Medium-term (1-2 months)
7. **Cross-Pane Intel Fusion**
   - Palm detects person → OPS Center highlights in Shadow Broker geospatial (if location match)
   - Shadow Broker flags unknown → OPS Center pulls gait profile
   - Single URL query string coordinates all three panes

8. **Skeleton Library Analytics**
   - Daily report: "5 unique skeletons detected, 3 matches to known profiles, 2 new unknowns"
   - Heatmap: arrival times by skeleton ID (pattern of life)
   - Stranger confidence scoring (gait anomaly + face mismatch + no known zone)

9. **Video Backfill from Tapo SD**
   - Already have backfill_tapo.py
   - Add: "backfill MoCap — last 7 days of doorbell clips"
   - Auto-process through FreeMoCap pipeline
   - Populate gait-vectors/ with historical data

---

## Technical Decisions to Lock

### FreeMoCap Invocation
- **Headless or REST?** The installer is Python. Likely: Python subprocess call with input/output paths.
- **Batch or streaming?** Batch for now (queue mocap-in/, process in background, output to mocap-out/).
- **GPU?** If available, auto-detect. Fallback to CPU (slower but works).
- **BVH format?** Standard humanoid BVH (17 joints). Store + render.

### Gait Vector Storage
- **Format:** CSV or JSON per skeleton? Or SQLite table?
- **Compression:** 18-dim float per frame. 30fps = 1800 values/min. Archive compressed after 7d.
- **Search:** Cosine distance in memory or quantize + LSH?

### OPS Center Integration
- **iFrame sandboxing:** Currently set to allow-same-origin, allow-scripts. Sufficient for local-only.
- **Cross-pane messaging:** Need PostMessage API if panes need to sync (e.g., Palm detects person → Shadow Broker highlights)?
- **Future:** Docker Compose for all three services + OPS Center proxy?

---

## Anti-Patterns to Avoid

1. **Don't average skeletons.** One skeleton = one identity. Multiple detections of the same person = multiple rows, matched by gait vector.

2. **Don't discard metadata.** Every skeleton needs timestamp, source camera, entry vector, exit vector. Queries depend on it.

3. **Don't process in real-time on first run.** Batch + queue pattern keeps the UI responsive. Background job finishes, webhook notifies dashboard.

4. **Don't lock the Seagate drive.** Overwrite policy: footage after 30d, mocap-in after successful processing, mocap-out after compression to archive.

---

## The Moat: Skeleton as Identity

Faces fail. Clothing changes. License plates spoof. **Skeleton doesn't lie.** 

The person who walked up to your door 47 times in the last 3 months has a unique gait signature. You can recognize them even if they wear different clothes, change their gate, or avoid the camera. Skeleton data is:
- **Durable** — survives wardrobe, angle, lighting changes
- **Persistent** — searchable across months of history
- **Forensic** — timestamps + positions = movement reconstruction
- **Community-safe** — skeleton is not a face, no privacy conflict

This is the feature that separates Palantir-grade home monitoring from consumer smart home cameras.

---

## Why Now?

You have:
1. ✓ Wired doorbell with RTSP
2. ✓ Storage (926GB Seagate)
3. ✓ FreeMoCap (just installed)
4. ✓ Unified dashboard (OPS Center v2 ready)
5. ✓ Gait engine already in Palm Command (gait_engine.py)

The pipeline is *almost* wired. MoCap is the missing link. Build it.

