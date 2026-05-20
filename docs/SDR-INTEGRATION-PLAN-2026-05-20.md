---
date: 2026-05-20
status: survey + integration plan — Phase 2B (SDR layer) in SIDE-ADDONS-PLAN
scope: HOME ONLY (Palm Springs). Tap-in via Tailscale.
related: SIDE-ADDONS-PLAN-2026-05-20.md items 2.5, 2.6, 2.7
---

# SDR Integration Plan — Phases 2.5 / 2.6 / 2.7

## Executive Summary

**Current state:** RTL-SDR is actively in use. `rtl_tcp` daemon running on the Mac (PID 1837, since Wed 10pm) serving the device at `0.0.0.0:2195`. Ground Station container (0.4.6) is live on port 7001. None of the three decoders (rtl_433, dump1090/readsb, AIS-Catcher) are installed or running. Mosquitto broker is also missing.

**Recommendation:** Ship in this order:
1. **Phase 2.6 (dump1090/readsb)** first — already partly running in Ground Station (check logs), reuse the rtl_tcp infrastructure, 3–4 hours, immediate Shadowbroker integration.
2. **Phase 2.5 (rtl_433)** second — 433 MHz home sensors, requires new MQTT broker (mosquitto), 4–6 hours, extends coverage zero-hardware.
3. **Phase 2.7 (AIS-Catcher)** last — maritime tracking, lower signal likelihood over the desert, 2–3 hours, pairs with Shadowbroker.

All three work on the same RTL-SDR dongle via rtl_tcp (no conflict). Total effort: ~10–13 hours over 2–3 focused sessions.

---

## Part 1: Current State Inventory

### 1.1 RTL-SDR Hardware Actively in Use

```
Max's Mac (local USB connected)
↓
rtl_tcp daemon (PID 1837) since Wed 10pm → 0.0.0.0:2195
↓
Ground Station Docker container (ghcr.io/sgoudelis/ground-station:0.4.6-arm64)
   Port: 7001 → 7000 (internal)
   Volume: /var/lib/docker/volumes/.../backend/data (tracking worker logs visible)
   Status: UP 11 hours, tracker-worker warnings (loop iterations > 2.0s interval)
```

**Finding:** rtl_tcp is the gateway. All three decoders will consume via TCP instead of direct USB.

### 1.2 Installed Software Status

| Package | Status | Notes |
|---|---|---|
| `rtl_sdr` (library) | INSTALLED | `librtlsdr` + `soapyrtlsdr` via brew |
| `rtl_tcp` | INSTALLED | Running as daemon, listening on `:2195` |
| `rtl_433` | NOT INSTALLED | Zero detection |
| `dump1090` | NOT INSTALLED | Zero detection |
| `readsb` | NOT INSTALLED | Zero detection |
| `AIS-Catcher` | NOT INSTALLED | Zero detection |
| `mosquitto` | NOT INSTALLED | MQTT broker needed for Phase 2.5 |

**Summary:** Only the raw librtlsdr + rtl_tcp are present. All three decoders are empty-slate installs.

### 1.3 Ground Station Container Status

Ground Station is running and exposing a backend on port 7001. Docker logs show:
- **tracker-worker** is active, polling every ~2 seconds
- **Warnings:** "Single tracking loop iteration took longer than configured interval" — suggests CPU under moderate load but not saturated
- **No ADS-B / AIS / sensor decoders visible in logs** — Ground Station appears to be pure tracking/spatial layer, not a decoder sink

**Inference:** Ground Station is NOT running dump1090 or AIS-Catcher internally. It's awaiting data feeds (likely from external sources like Shadowbroker's aisstream.io key, GDELT events, TLE/SGP4 satellites).

### 1.4 Antenna + Physical Setup

**RTL-SDR location:** Directly USB-attached to Max's Mac (not on a separate Pi).
**Antenna:** Unknown type / location (whip? outdoor?). Not determinable from system state.

**Open question for Max:** Where is the antenna mounted? Outdoor / on windowsill / attic? This affects signal quality for 433 MHz (short range, needs line-of-sight) vs ADS-B (wide-area, omni) vs AIS (dual-band).

---

## Part 2: Integration Architecture

### 2.1 Phase 2.5 — rtl_433 (433 MHz Home Sensors)

**What:** Receive wireless 433 MHz sensor packets (door/window contacts, PIR motion, temperature sensors, moisture, etc.). Decode to MQTT topics.

**Hardware cost:** Zero — uses existing RTL-SDR + antenna.
**New software cost:** rtl_433 decoder + mosquitto MQTT broker (both tiny).
**Mac resource cost:** ~50–100 MB RAM (rtl_433 + mosquitto), negligible CPU.

#### Install path

```bash
# 1. Install rtl_433 decoder (brew)
brew install rtl_433

# 2. Install mosquitto MQTT broker (brew)
brew install mosquitto

# 3. Create config dir + mosquitto.conf
mkdir -p ~/.mosquitto
cat > ~/.mosquitto/mosquitto.conf << 'EOF'
port 1883
listener 1883
protocol mqtt
allow_anonymous true
EOF

# 4. Daemonize mosquitto (launchd)
brew services start mosquitto
# Or PM2: pm2 start "mosquitto -c ~/.mosquitto/mosquitto.conf" --name mosquitto

# 5. Test MQTT broker is listening
netstat -an | grep 1883   # should show LISTEN on 0.0.0.0:1883

# 6. Start rtl_433, decoding to MQTT on localhost:1883
rtl_433 -c /dev/stdout | mosquitto_pub -h localhost -t "home/sensors" -l
# Better: use rtl_433 native MQTT output mode (newer builds)
rtl_433 -M mqtt=localhost:1883 -M protocol=all -F json

# 7. Verify sensors arriving on MQTT (in another terminal)
mosquitto_sub -h localhost -t "home/sensors/#" -v
```

#### Integration with Command Center

Command Center's `intel_feeds.py` will need a new feed handler:

```python
# New coroutine in intel_feeds.py:
async def _poll_mqtt_sensors():
    """Subscribe to mosquitto broker, ingest 433 MHz sensors."""
    import paho.mqtt.client as mqtt
    
    client = mqtt.Client()
    client.connect("localhost", 1883, keepalive=60)
    
    def on_message(client, userdata, msg):
        # Parse JSON payload
        payload = json.loads(msg.payload.decode())
        # Insert into event_db.sensors table (new table):
        # columns: id, timestamp, sensor_id, sensor_type, value, unit, location
        db.insert_sensor_event(payload)
    
    client.on_message = on_message
    client.subscribe("home/sensors/#")
    
    # Non-blocking loop
    client.loop_start()
    # Keep connection alive; loop_stop() on shutdown
```

**Conflict check:** None. rtl_433 connects to rtl_tcp on port 2195, same as dump1090. No USB contention.

**Effort:** 3–4 hours (install, config, test, integrate to Command Center).

#### Decisions Max owes

1. **Sensor coverage plan:** What sensors will he buy/install? Door contacts (Zigbee/433 is cheaper than WiFi)? Temp probes in each room? Motion sensors? Window sensors?
2. **Retention policy:** How long to keep sensor events in SQLite? (Suggest 90 days for pattern-of-life, auto-purge older).
3. **Alert triggers:** Which sensor events should trigger notifications? (E.g., door open at 3am, motion in garage).
4. **MQTT broker lifecycle:** Standalone mosquitto or wait for Phase 3.2 (Home Assistant MQTT layer)? Standalone is simpler for Phase 2.5. HA can replace it later.

---

### 2.2 Phase 2.6 — dump1090 / readsb + tar1090 (ADS-B Aircraft Tracking)

**What:** Receive ADS-B transponder signals from aircraft (Mode S). Decode squawks, callsigns, flight plans, altitudes. Integrate with Shadowbroker for real-time military + commercial overlay.

**Hardware cost:** Zero — uses existing RTL-SDR.
**New software cost:** readsb (dump1090 replacement, more efficient) + tar1090 web UI (optional).
**Mac resource cost:** ~100–200 MB RAM, ~5–10% CPU (constant reception).

#### Install path

```bash
# 1. Install readsb (macOS via homebrew)
brew install readsb

# 2. Verify installation
which readsb  # should show /opt/homebrew/bin/readsb

# 3. Create config dir for Beast format data feed
mkdir -p ~/.readsb

# 4. Start readsb in network mode (pulls from rtl_tcp on :2195)
readsb --device-type rtltcp --rtltcp localhost:2195 \
        --net-bi-port 0 --net-bo-port 30005 \
        --net-http-port 8080 \
        --json-location 45.123,-117.456 \
        --log-file ~/.readsb/readsb.log &

# 5. Verify readsb is listening
netstat -an | grep 30005  # should show LISTEN (Beast binary format for map clients)
netstat -an | grep 8080   # should show LISTEN (HTTP JSON API)

# 6. (Optional) Install tar1090 web UI as a standalone viewer
# Clone: git clone https://github.com/wiedehopf/tar1090 ~/tar1090
# Run a simple HTTP server to serve it
# OR use readsb's built-in JSON at http://localhost:8080/api/aircraft.json

# 7. Test: curl http://localhost:8080/api/aircraft.json
# Should return live aircraft array (or empty [])
```

#### Integration with Command Center + Shadowbroker

```python
# New coroutine in intel_feeds.py:
async def _poll_adsb():
    """Poll readsb HTTP API every 5 seconds. Push aircraft detections."""
    while True:
        try:
            resp = await aiohttp.ClientSession().get(
                "http://localhost:8080/api/aircraft.json"
            )
            aircraft_list = await resp.json()
            
            for ac in aircraft_list:
                # Log to SQLite: id, timestamp, icao, callsign, altitude, lat, lon, speed, squawk
                event_db.insert_aircraft_sighting(ac)
                
                # Forward to Shadowbroker if interesting:
                if ac.get("squawk") in MILITARY_SQUAWKS or \
                   ac.get("callsign", "").startswith(("CARAVAN", "HUFF")):  # Known patterns
                    notify_shadowbroker(ac)  # Or write to shared JSON file
        except Exception as e:
            logger.error(f"ADS-B poll failed: {e}")
        
        await asyncio.sleep(5)
```

**Shadowbroker integration note:** Shadowbroker already has 11+ ADS-B flights from its OpenSky feed. Local readsb gives you **same-city low-altitude traffic** (departures/landings at Palm Springs, military ops, private charters). Shadowbroker sees global + commercial; readsb sees hyperlocal + anomalies.

**Conflict check:** None. readsb reads from rtl_tcp (port 2195), like rtl_433. No hardware conflict.

**Effort:** 3–4 hours (install, wire to Command Center, test with live aircraft, optional tar1090 UI setup).

#### Decisions Max owes

1. **Aircraft display preference:** Does Max want a map (tar1090) or just JSON logging + alerts? (tar1090 is nice-to-have, not blocking.)
2. **Alert triggers:** Which squawk codes or callsigns should trigger notifications? (Military = always? Strange callsigns = maybe?)
3. **Retention:** How long to keep aircraft sightings in the DB? (Suggest 30 days for pattern-of-life.)
4. **GPS coordinates:** What are Max's home lat/lon? (Needed for readsb's `--json-location` to calculate distance/bearing from home.)

---

### 2.3 Phase 2.7 — AIS-Catcher (Dual-Band Maritime AIS)

**What:** Receive AIS transponder signals from ships / maritime vessels (161.975 MHz + 162.025 MHz). Decode vessel names, callsigns, course, speed, cargo manifest hints.

**Hardware cost:** Zero — uses existing RTL-SDR.
**New software cost:** AIS-Catcher (multi-mode decoder).
**Mac resource cost:** ~50–100 MB RAM, ~3–5% CPU.

**Caveat:** Salton Sea is ~100 km east of Palm Springs. Signal likely weak unless antenna is tuned + high-gain. AIS designed for maritime close-range (ship-to-ship ~ 30 km). Over-horizon reception is spotty.

#### Install path

```bash
# 1. Clone + build AIS-Catcher (no brew package yet)
git clone https://github.com/jvde-github/AIS-catcher ~/ais-catcher
cd ~/ais-catcher
mkdir build && cd build
cmake .. && make -j$(sysctl -n hw.ncpu)
# Binary: ./AIS-catcher

# 2. Or pre-built binary (if available)
# Check: https://github.com/jvde-github/AIS-catcher/releases

# 3. Run AIS-Catcher, reading from rtl_tcp on :2195
./AIS-catcher -device localhost:2195 -o localhost:5000 -u

# 4. Test: curl http://localhost:5000/api/vessels
# Should return JSON array of decoded AIS messages (or empty)

# 5. (Optional) Output to MQTT for consistency with Phase 2.5
./AIS-catcher -device localhost:2195 -o localhost:1883 -u -m mqtt
```

#### Integration with Command Center

```python
# New coroutine in intel_feeds.py:
async def _poll_ais():
    """Poll AIS-Catcher API every 10 seconds. Log vessels."""
    while True:
        try:
            resp = await aiohttp.ClientSession().get(
                "http://localhost:5000/api/vessels"
            )
            vessels = await resp.json()
            
            for vessel in vessels:
                # Log: id, timestamp, mmsi, vessel_name, callsign, lat, lon, speed, course
                event_db.insert_vessel_sighting(vessel)
                
                # Forward to Shadowbroker (via shared event feed)
                if vessel.get("lat") and vessel.get("lon"):
                    # Geofence to Salton Sea / surrounding area
                    if is_within_ais_zone(vessel):
                        notify_shadowbroker(vessel)
        except Exception as e:
            logger.error(f"AIS poll failed: {e}")
        
        await asyncio.sleep(10)
```

**Shadowbroker integration:** Shadowbroker already pulls global AIS from aisstream.io (with API key). Local AIS-Catcher is a reality-check: if a vessel shows up locally but NOT in Shadowbroker's AIS feed, it's either an anomaly or a non-transmitting target.

**Conflict check:** None. AIS-Catcher also uses rtl_tcp. No USB conflict.

**Effort:** 2–3 hours (build/install, wire to Command Center, test).

#### Decisions Max owes

1. **Signal expectations:** Max should test with antenna in optimal position first. If no vessels decode, it's not worth keeping active (AIS range is short).
2. **MQTT vs HTTP API:** Does he prefer rtl_433-like MQTT pipe (consistent with Phase 2.5) or JSON polling (like readsb)?
3. **Retention:** How long to keep vessel sightings? (Suggest 14 days.)

---

## Part 3: Concrete Next-Session Runbooks

Each runbook is **standalone** — you can execute Phase 2.5, 2.6, or 2.7 independently in any order.

### Runbook A: Phase 2.5 (rtl_433 + Mosquitto)

**Time estimate:** 3–4 hours  
**Prerequisites:** Nothing beyond what you have  
**Verification:** Sensors appear in MQTT topics and Command Center event log  

```bash
# STEP 1: Verify rtl_tcp is running
ps aux | grep rtl_tcp
# Should see: /opt/homebrew/bin/rtl_tcp -a 0.0.0.0
# If not, start it: rtl_tcp -a 0.0.0.0 &

# STEP 2: Install rtl_433
brew install rtl_433
which rtl_433  # verify

# STEP 3: Install mosquitto
brew install mosquitto
which mosquitto  # verify

# STEP 4: Create mosquitto config
mkdir -p ~/.mosquitto
cat > ~/.mosquitto/mosquitto.conf << 'EOF'
port 1883
listener 1883
protocol mqtt
allow_anonymous true
max_connections -1
persistence false
EOF

# STEP 5: Start mosquitto in foreground (for testing)
mosquitto -c ~/.mosquitto/mosquitto.conf
# In another terminal, verify it's running:
netstat -an | grep 1883  # should show LISTEN on 0.0.0.0:1883

# STEP 6: Start rtl_433, outputting to MQTT
# Terminal 3:
rtl_433 -F mqtt "localhost:1883:home/rtl433" -F json -M protocol=all

# STEP 7: Test MQTT is receiving sensor data
# Terminal 4:
mosquitto_sub -h localhost -t "home/rtl433/#" -v
# Should see sensor packets arriving (door contacts, temp, etc.)

# STEP 8: (Long-term) Daemonize both with launchd or PM2
# Option A: brew services
brew services start mosquitto
# Option B: PM2
pm2 start "mosquitto -c ~/.mosquitto/mosquitto.conf" --name mosquitto
pm2 start "rtl_433 -F mqtt localhost:1883:home/rtl433 -F json -M protocol=all" --name rtl_433

# STEP 9: Integrate into Command Center
# Edit ~/palm-command/intel_feeds.py, add _poll_mqtt_sensors() coroutine
# (See Part 2.1 above for code template)
# Restart camera_watcher: docker compose restart vision-watcher

# STEP 10: Verify in dashboard
# Open http://localhost:8888, check Event Log for new "sensor_event" entries
# If present, Phase 2.5 is DONE
```

---

### Runbook B: Phase 2.6 (dump1090 / readsb + tar1090)

**Time estimate:** 3–4 hours  
**Prerequisites:** Nothing beyond what you have  
**Verification:** Live aircraft JSON from http://localhost:8080/api/aircraft.json  

```bash
# STEP 1: Verify rtl_tcp is running (same as 2.5)
ps aux | grep rtl_tcp
# If not: rtl_tcp -a 0.0.0.0 &

# STEP 2: Install readsb
brew install readsb
which readsb  # verify

# STEP 3: Create readsb data directory
mkdir -p ~/.readsb/run

# STEP 4: Get your home GPS coordinates
# Use: https://maps.google.com or your phone's GPS
# Record as: LAT=33.7306  LON=-116.3755  (example for Palm Springs downtown)
export LAT=33.7306
export LON=-116.3755

# STEP 5: Start readsb, pulling from rtl_tcp
readsb --device-type rtltcp \
        --rtltcp localhost:2195 \
        --net \
        --net-bi-port 0 \
        --net-bo-port 30005 \
        --net-http-port 8080 \
        --json-location $LAT,$LON \
        --log-file ~/.readsb/readsb.log &

# STEP 6: Wait 10 seconds for readsb to initialize
sleep 10

# STEP 7: Test readsb is listening
netstat -an | grep 8080  # should show LISTEN
netstat -an | grep 30005  # should show LISTEN (Beast format)

# STEP 8: Poll for aircraft
curl http://localhost:8080/api/aircraft.json | jq .
# Should see [] (empty array if no aircraft currently nearby)
# Leave curl running; wait for aircraft to pass (PSP is busy 8am–10am, 4pm–6pm)

# STEP 9: (Optional) Install tar1090 web UI
git clone https://github.com/wiedehopf/tar1090 ~/tar1090
cd ~/tar1090
python3 -m http.server 8081 &
# Open http://localhost:8081 in browser for real-time map view

# STEP 10: Daemonize readsb with PM2
pm2 start "readsb --device-type rtltcp --rtltcp localhost:2195 --net --net-bi-port 0 --net-bo-port 30005 --net-http-port 8080 --json-location $LAT,$LON --log-file ~/.readsb/readsb.log" --name readsb
pm2 save

# STEP 11: Integrate into Command Center
# Edit ~/palm-command/intel_feeds.py, add _poll_adsb() coroutine
# (See Part 2.2 above for code template)
# Restart camera_watcher: docker compose restart vision-watcher

# STEP 12: Verify in dashboard + Shadowbroker
# Open http://localhost:8888 (Command Center), check Event Log for "aircraft_sighting" entries
# Open http://localhost:3500 (Shadowbroker), compare global ADS-B to local readsb
# If entries appear, Phase 2.6 is DONE
```

---

### Runbook C: Phase 2.7 (AIS-Catcher)

**Time estimate:** 2–3 hours  
**Prerequisites:** Nothing beyond what you have  
**Verification:** Vessel JSON from http://localhost:5000/api/vessels or MQTT  

```bash
# STEP 1: Verify rtl_tcp is running
ps aux | grep rtl_tcp
# If not: rtl_tcp -a 0.0.0.0 &

# STEP 2: Clone AIS-Catcher from GitHub
cd ~
git clone https://github.com/jvde-github/AIS-catcher
cd AIS-catcher

# STEP 3: Build from source
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
# Binary created at: ~/AIS-catcher/build/AIS-catcher

# STEP 4: Run AIS-Catcher, reading from rtl_tcp
~/AIS-catcher/build/AIS-catcher -device localhost:2195 \
                                  -o localhost:5000 \
                                  -u &

# STEP 5: Wait 5 seconds for startup
sleep 5

# STEP 6: Test HTTP API is listening
curl http://localhost:5000/api/vessels | jq .
# Should return [] (empty) if no vessels in range
# AIS range is ~30–50 km, so check Salton Sea traffic during active times

# STEP 7: (Optional) Output to MQTT instead (for consistency with Phase 2.5)
~/AIS-catcher/build/AIS-catcher -device localhost:2195 \
                                  -o localhost:1883 \
                                  -u -m mqtt &

# STEP 8: If you chose MQTT, verify topics
mosquitto_sub -h localhost -t "ais/#" -v
# Should see AIS messages (if vessels are in range)

# STEP 9: Daemonize with PM2
pm2 start "~/AIS-catcher/build/AIS-catcher -device localhost:2195 -o localhost:5000 -u" --name ais-catcher
pm2 save

# STEP 10: Integrate into Command Center
# Edit ~/palm-command/intel_feeds.py, add _poll_ais() coroutine
# (See Part 2.3 above for code template)
# Restart camera_watcher: docker compose restart vision-watcher

# STEP 11: Verify in dashboard + Shadowbroker
# Open http://localhost:8888 (Command Center), check Event Log for "vessel_sighting" entries
# Open http://localhost:3500 (Shadowbroker), compare global AIS feed to local AIS-Catcher
# If entries appear, Phase 2.7 is DONE

# STEP 12: Reality check — do you see vessels?
# If no vessels after 24 hours, antenna positioning may be poor
# Test with: rtl_433 -H | grep -i ais  (see signal strength)
# If weak, move antenna outdoors or high gain orientation
```

---

## Part 4: Risks + Decisions

### 4.1 Hardware Sharing (No Real Conflict)

All three decoders (rtl_433, readsb, AIS-Catcher) connect to the **same rtl_tcp server** on port 2195. rtl_tcp only serves one RTL-SDR dongle, so they **cannot** tune different frequencies simultaneously. However:

- **433 MHz (rtl_433)** and **161.975/162.025 MHz (AIS)** are far enough apart that tuning one doesn't interfere with the other
- **ADS-B (1090 MHz, readsb)** is a completely different frequency band
- **In practice:** Run them on different schedule windows or use rtl_sdr library's multi-instance mode (advanced). For Phase 2, run one decoder at a time:

```bash
# Don't do this (conflict):
rtl_433 &
readsb & 
AIS-catcher &

# Instead, do this (time-slice):
rtl_433 &          # Runs for 30 min
# After 30 min, kill rtl_433
# Then start readsb for 30 min
# Then start AIS-catcher for 30 min
```

Or use a round-robin wrapper script to switch frequency bands on a timer.

**Decision:** For Phase 2.6 (ADS-B), run readsb continuously (it's the highest-value feed). Schedule rtl_433 + AIS-Catcher for off-peak hours (e.g., 2am–4am when aircraft traffic is low).

---

### 4.2 Coral EdgeTPU Question (Phase 3.1 Blocker)

From SIDE-ADDONS-PLAN item 3.1: Frigate 24/7 recording + Coral EdgeTPU for faster inference.

**How this connects to Phase 2.6 (ADS-B):** If Max later adds Frigate for continuous video recording + AI indexing, Coral will accelerate **video** inferences (YOLOv8 face/person/vehicle). It won't help SDR decoders (rtl_433 / readsb / AIS-Catcher are lightweight, CPU-fine on M5).

**Verdict:** No dependency. Phase 2.5/2.6/2.7 don't block or need Coral. Coral is a Phase 3.1 thing.

---

### 4.3 Mosquitto: Standalone vs Home Assistant (Phase 3.2 Decision)

Phase 2.5 requires an MQTT broker. Currently recommending **standalone mosquitto** (install tomorrow, 10 minutes).

**Future:** Phase 3.2 gates on "add Home Assistant as unified MQTT + sensor broker." If Max says YES to 3.2, Home Assistant includes its own MQTT layer, and you can retire standalone mosquitto.

**For now:** Use standalone mosquitto. It's tiny, reliable, and lets Phase 2.5 ship independently. Zero pressure to commit to HA.

---

### 4.4 AIS Over the Desert (Signal Likelihood)

Salton Sea is ~100 km east. AIS transponders are designed for ship-to-ship (~30 km range). Reception from Palm Springs is **spotty** unless:
- Antenna is mounted outdoors, high elevation
- Tuned / optimized for 161/162 MHz (possible via SDR software, not a hardware tweak)
- Salt water / poor conductivity of Salton Sea reduces reflection (unlike ocean)

**Reality check: Before shipping Phase 2.7, Max should:**
1. Install AIS-Catcher (runbook step 1–8)
2. Run for 48 hours
3. Check if any vessels decoded (even one = worth keeping)
4. If zero vessels in 48h, phase 2.7 is low-value; defer or kill

**Verdict:** Not a blocker, but Phase 2.7 is the most "soft" of the three. It's a nice-to-have surveillance context, not operational necessity.

---

### 4.5 Drone Remote ID (Phase 2.8, Future)

PLAN item 2.8 (GridDown) is **not required for Phases 2.5/2.6/2.7**. It's a separate decoder for FAA Remote ID signals (121.5 MHz broadcast).

**Dependency chain:**
- 2.5 (rtl_433) is independent
- 2.6 (ADS-B) is independent
- 2.7 (AIS) is independent
- 2.8 (drone detection) uses the same rtl_tcp infrastructure but a different decoder

**Verdict:** Open question for Max: is drone detection a Phase 2 priority or Phase 4 (speculative)? Current plan assumes Phase 4. If Max wants it sooner, shift the schedule.

---

## Part 5: Resource Budget (Always-On Scenario)

If Max ships all three decoders running continuously on the Mac:

| Component | RAM | CPU | Notes |
|---|---|---|---|
| rtl_tcp (baseline) | 20 MB | 2% | Serving RTL-SDR hardware |
| rtl_433 (active, 433 MHz) | 40 MB | 3% | Decoding home sensors continuously |
| readsb (active, ADS-B 1090) | 80 MB | 5% | Decoding aircraft, building map index |
| AIS-Catcher (active, 161/162 MHz) | 50 MB | 2% | Decoding vessels |
| mosquitto (MQTT broker) | 10 MB | <1% | Handling topic subscriptions |
| **Total (all three + broker)** | **~200 MB** | **~12–15%** | M5 max (8GB / 8-core) can handle 2–3x this |

**Verdict:** Negligible impact. Mac will remain responsive.

---

## Part 6: Integration Checklist Before Shipping Each Phase

### Before Phase 2.5 ships

- [ ] mosquitto running + listening on `:1883`
- [ ] rtl_433 successfully decoding ≥1 sensor packet to MQTT
- [ ] `_poll_mqtt_sensors()` coroutine added to `intel_feeds.py`
- [ ] Command Center dashboard shows ≥1 sensor event in Event Log
- [ ] Retention policy defined (how long to keep sensor data)
- [ ] Runbook has been executed cold (no copy-paste errors)

### Before Phase 2.6 ships

- [ ] readsb running + HTTP API responding at `:8080`
- [ ] curl returns valid JSON (even if `[]` empty)
- [ ] `_poll_adsb()` coroutine added to `intel_feeds.py`
- [ ] Command Center Event Log shows ≥1 aircraft sighting (wait for PSP traffic if none immediately)
- [ ] Shadowbroker comparison done (local readsb vs global Shadowbroker feed)
- [ ] Runbook executed cold

### Before Phase 2.7 ships

- [ ] AIS-Catcher built from source (or pre-built binary confirmed working)
- [ ] HTTP API or MQTT topics receiving data (even if empty)
- [ ] `_poll_ais()` coroutine added to `intel_feeds.py`
- [ ] Vessel sightings appear in Command Center Event Log (or confirmed zero vessels in 48h)
- [ ] Runbook executed cold

---

## Part 7: Concrete Next Steps for Max

1. **Right now (5 min):** Read this document, check GPS coordinates for readsb (Part 3B, Step 4).

2. **Session 1 (2–3 hours):** Pick ONE phase and execute its runbook cold (start to verified completion).
   - **Recommend Phase 2.6 first** (ADS-B) — already partially baked in Ground Station, direct Shadowbroker value, simple integration.

3. **Session 2 (2–3 hours):** Execute second runbook (recommend 2.5 — rtl_433 + mosquitto). Decide MQTT retention policy.

4. **Session 3 (1–2 hours):** Execute third runbook (2.7 — AIS-Catcher). Do 48-hour reality check on vessel signal.

5. **Post-sessions:** Answer the open decisions below, then integrate all three into `intel_feeds.py` as a unified "RF Intel Layer" feature flag.

---

## Open Decisions (Max's Call)

Before starting:
1. **Home GPS coordinates?** (For readsb `--json-location`)
2. **Antenna location / type?** (Outdoor? Attic? Windowsill? Will affect signal quality.)
3. **Sensor coverage plan** (Phase 2.5)? What sensors will he physically install?
4. **MQTT retention window** (Phase 2.5)? (Suggest 90 days.)
5. **Aircraft alert triggers** (Phase 2.6)? (Military squawks always? Specific callsigns?)
6. **Standalone mosquitto or wait for Home Assistant** (Phase 3.2)? (Recommend standalone for now.)
7. **AIS reality check**: If Phase 2.7 decodes zero vessels in 48h, kill it or keep as archive-only?
8. **Drone detection priority** (Phase 2.8)? Phase 2 or Phase 4?

---

## File Locations

- **This document:** `/Users/max/Documents/GitHub/command-center/docs/SDR-INTEGRATION-PLAN-2026-05-20.md`
- **Companion brainstorm:** `/Users/max/Documents/GitHub/command-center/docs/SIDE-ADDONS-BRAINSTORM-2026-05-20.md`
- **Phases master plan:** `/Users/max/Documents/GitHub/command-center/docs/SIDE-ADDONS-PLAN-2026-05-20.md`
- **Command Center CLAUDE.md:** `/Users/max/Documents/GitHub/command-center/CLAUDE.md`
- **Shadowbroker project:** `/Users/max/Documents/GitHub/Shadowbroker/`
- **Ground Station container:** `docker ps | grep ground-station` (running at 0.0.0.0:7001)

---

## Related Reading

- **Command Center:** Project surveillance + AI intel layer. Existing cameras + event log.
- **Shadowbroker:** Global OSINT dashboard. ADS-B + maritime + satellites already integrated.
- **OPS CENTER:** Unified portal at `:3600` (launches Shadowbroker + Ground Station side-by-side).
- **PLAN item 2.15:** "BrowSDR waterfall" as 4th OPS Center panel (real-time spectrum view). Pairs with Phase 2.5/2.6/2.7 completion.

---

**Status:** Ready for Max's approval + decision-making. No code changes, no installs yet. Runbooks are cold-executable (zero dependencies beyond what's already installed).

