#!/usr/bin/env python3.12
"""
PALM COMMAND — External Intelligence Feeds Engine

Real-time threat awareness from public APIs — no API keys required.

Sources:
  NWS     — National Weather Service: weather + fire weather alerts
  USGS    — Seismic activity (Coachella Valley is on the San Andreas fault)
  CALFIRE — Active wildfire incidents statewide (filtered to region)
  Citizen — Hyperlocal 911-sourced incidents (crime, fire, EMS, accidents)
  FEMA    — IPAWS emergency alerts

All feeds are cached and refreshed on background threads.
Designed to run standalone or be imported by camera_watcher.py.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

# ── Home location (override via env vars) ────────────────────────
HOME_LAT  = float(os.environ.get("HOME_LAT",  "33.8303"))
HOME_LON  = float(os.environ.get("HOME_LON", "-116.5453"))
HOME_ZIP  = os.environ.get("HOME_ZIP",  "92262")
HOME_NAME = os.environ.get("HOME_NAME", "Palm Springs, CA")

# Radius filters (km)
QUAKE_RADIUS_KM  = float(os.environ.get("QUAKE_RADIUS_KM",  "200"))
FIRE_RADIUS_KM   = float(os.environ.get("FIRE_RADIUS_KM",   "120"))
CRIME_RADIUS_DEG = 0.5  # roughly 55km lat/lon degrees

# Cache TTLs (seconds)
TTL_WEATHER  = 300   # 5 min
TTL_QUAKE    = 120   # 2 min — seismic can evolve rapidly
TTL_FIRE     = 600   # 10 min
TTL_CITIZEN  = 180   # 3 min
TTL_ALL      = 60    # combined feed min refresh

_USER_AGENT = "PALM-COMMAND/2.0 (home-security; contact=local)"

# ── Data types ────────────────────────────────────────────────────

@dataclass
class FeedItem:
    source:    str           # "NWS" | "USGS" | "CALFIRE" | "CITIZEN"
    severity:  str           # "RED" | "ORANGE" | "YELLOW" | "GREEN" | "INFO"
    category:  str           # "WEATHER" | "SEISMIC" | "FIRE" | "CRIME" | "EMS" | "POWER" | ...
    title:     str
    detail:    str
    location:  str
    ts:        float         # unix timestamp
    distance_km: Optional[float] = None
    lat:       Optional[float]   = None
    lon:       Optional[float]   = None
    url:       Optional[str]     = None
    raw:       dict              = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source":      self.source,
            "severity":    self.severity,
            "category":    self.category,
            "title":       self.title,
            "detail":      self.detail,
            "location":    self.location,
            "ts":          self.ts,
            "ts_human":    datetime.fromtimestamp(self.ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "distance_km": round(self.distance_km, 1) if self.distance_km else None,
            "lat":         self.lat,
            "lon":         self.lon,
            "url":         self.url,
        }


# ── Utility ───────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def _fetch(url: str, timeout: int = 10, headers: dict | None = None) -> dict | list | None:
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
            **(headers or {}),
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"[feeds] fetch error {url[:60]}: {e}", flush=True)
        return None


def _nws_severity(sev: str, urgency: str, certainty: str) -> str:
    sev = (sev or "").lower()
    if sev in ("extreme", "severe"):
        return "RED"
    if sev == "moderate":
        return "ORANGE"
    if urgency and "immediate" in urgency.lower():
        return "ORANGE"
    if sev == "minor":
        return "YELLOW"
    return "INFO"


# ── Feed Cache ────────────────────────────────────────────────────

@dataclass
class _Cache:
    items: list[FeedItem] = field(default_factory=list)
    last_ts: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def stale(self, ttl: float) -> bool:
        return (time.time() - self.last_ts) > ttl

    def update(self, items: list[FeedItem]):
        with self.lock:
            self.items = items
            self.last_ts = time.time()

    def get(self) -> list[FeedItem]:
        with self.lock:
            return list(self.items)


_cache_weather  = _Cache()
_cache_quake    = _Cache()
_cache_fire     = _Cache()
_cache_citizen  = _Cache()


# ── NWS Weather + Fire Alerts ─────────────────────────────────────

def fetch_weather_alerts() -> list[FeedItem]:
    """
    Pull NWS active alerts for the home coordinates.
    Includes: red flag warnings, fire weather watches, extreme heat, air quality.
    """
    url = f"https://api.weather.gov/alerts/active?point={HOME_LAT},{HOME_LON}"
    data = _fetch(url)
    if not data:
        return []

    items: list[FeedItem] = []
    for feat in data.get("features", []):
        p = feat.get("properties", {})
        event     = p.get("event", "Unknown Alert")
        sev       = p.get("severity", "")
        urgency   = p.get("urgency", "")
        certainty = p.get("certainty", "")
        headline  = p.get("headline", event)
        desc      = (p.get("description", "") or "")[:300].replace("\n", " ")
        onset_s   = p.get("onset") or p.get("effective") or ""
        areas     = p.get("areaDesc", HOME_NAME)

        # Parse onset timestamp
        try:
            ts = datetime.fromisoformat(onset_s.replace("Z", "+00:00")).timestamp()
        except Exception:
            ts = time.time()

        # Category mapping
        ev_l = event.lower()
        if any(w in ev_l for w in ("fire", "red flag", "smoke")):
            cat = "FIRE_WEATHER"
        elif any(w in ev_l for w in ("heat", "temperature")):
            cat = "EXTREME_HEAT"
        elif "air quality" in ev_l or "smoke" in ev_l:
            cat = "AIR_QUALITY"
        elif any(w in ev_l for w in ("dust", "wind", "haboob")):
            cat = "WIND_DUST"
        elif any(w in ev_l for w in ("flood", "flash")):
            cat = "FLOOD"
        elif any(w in ev_l for w in ("earthquake", "tsunami")):
            cat = "SEISMIC"
        else:
            cat = "WEATHER"

        items.append(FeedItem(
            source   = "NWS",
            severity = _nws_severity(sev, urgency, certainty),
            category = cat,
            title    = event,
            detail   = headline if headline != event else desc[:200],
            location = areas,
            ts       = ts,
            distance_km = 0.0,  # NWS alerts are for our zone
            url      = p.get("@id"),
            raw      = {"severity": sev, "urgency": urgency, "certainty": certainty},
        ))

    _cache_weather.update(items)
    return items


def get_weather_alerts() -> list[dict]:
    if _cache_weather.stale(TTL_WEATHER):
        fetch_weather_alerts()
    return [i.to_dict() for i in _cache_weather.get()]


# ── USGS Seismic Monitor ─────────────────────────────────────────

def fetch_earthquakes() -> list[FeedItem]:
    """
    USGS real-time earthquake feed for Coachella Valley region.
    The area sits astride the San Andreas and associated fault zones.
    """
    url = (
        f"https://earthquake.usgs.gov/fdsnws/event/1/query"
        f"?format=geojson"
        f"&latitude={HOME_LAT}&longitude={HOME_LON}"
        f"&maxradiuskm={int(QUAKE_RADIUS_KM)}"
        f"&minmagnitude=1.5"
        f"&limit=25"
        f"&orderby=time"
    )
    data = _fetch(url)
    if not data:
        return []

    items: list[FeedItem] = []
    for feat in data.get("features", []):
        p      = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [None, None, None])
        mag    = p.get("mag", 0) or 0
        place  = p.get("place", "Unknown location")
        ts_ms  = p.get("time", 0) or 0
        status = p.get("status", "")
        depth  = coords[2] or 0

        lat = coords[1]
        lon = coords[0]
        dist_km = _haversine_km(HOME_LAT, HOME_LON, lat, lon) if lat and lon else None

        # Severity by magnitude
        if mag >= 5.0:
            sev = "RED"
        elif mag >= 4.0:
            sev = "ORANGE"
        elif mag >= 3.0:
            sev = "YELLOW"
        else:
            sev = "INFO"

        detail = f"M{mag:.1f} at depth {depth:.0f}km · {place}"
        if dist_km:
            detail += f" · {dist_km:.0f}km from home"

        items.append(FeedItem(
            source      = "USGS",
            severity    = sev,
            category    = "SEISMIC",
            title       = f"M{mag:.1f} Earthquake — {place}",
            detail      = detail,
            location    = place,
            ts          = ts_ms / 1000,
            distance_km = dist_km,
            lat         = lat,
            lon         = lon,
            url         = p.get("url"),
            raw         = {"mag": mag, "depth": depth, "status": status},
        ))

    _cache_quake.update(items)
    return items


def get_earthquakes() -> list[dict]:
    if _cache_quake.stale(TTL_QUAKE):
        fetch_earthquakes()
    return [i.to_dict() for i in _cache_quake.get()]


# ── CAL FIRE Active Incidents ─────────────────────────────────────

_CALFIRE_COUNTIES = {
    "riverside", "san bernardino", "san diego", "imperial",
    "los angeles", "orange", "ventura", "kern",
}

def fetch_fire_incidents() -> list[FeedItem]:
    """
    CAL FIRE active incident list. Filtered to Southern California counties.
    Includes acreage, containment %, and location.
    """
    url = "https://incidents.fire.ca.gov/umbraco/api/IncidentApi/List?inactive=false"
    data = _fetch(url, timeout=12)
    if not data or not isinstance(data, list):
        return []

    items: list[FeedItem] = []
    for inc in data:
        county    = (inc.get("County") or "").lower()
        name      = inc.get("Name") or "Unknown Fire"
        acres     = inc.get("AcresBurned") or 0
        contained = inc.get("PercentContained") or 0
        admin_u   = inc.get("AdminUnit") or ""
        started   = inc.get("Started") or ""
        lat_s     = inc.get("Latitude")
        lon_s     = inc.get("Longitude")
        url_link  = inc.get("Url") or ""

        # Filter to our region
        if county and county not in _CALFIRE_COUNTIES:
            continue

        lat = float(lat_s) if lat_s else None
        lon = float(lon_s) if lon_s else None
        dist_km = _haversine_km(HOME_LAT, HOME_LON, lat, lon) if lat and lon else None

        if dist_km and dist_km > FIRE_RADIUS_KM:
            continue

        # Parse timestamp
        try:
            ts = datetime.fromisoformat(started.replace("Z", "+00:00")).timestamp()
        except Exception:
            ts = time.time()

        # Severity by size and containment
        if acres >= 1000 and contained < 50:
            sev = "RED"
        elif acres >= 100 and contained < 75:
            sev = "ORANGE"
        elif acres >= 10:
            sev = "YELLOW"
        else:
            sev = "INFO"

        county_title = county.title() if county else "SoCal"
        detail = f"{acres:,.0f} acres · {contained}% contained · {county_title} County"
        if dist_km:
            detail += f" · {dist_km:.0f}km from home"
        if admin_u:
            detail += f" [{admin_u}]"

        items.append(FeedItem(
            source      = "CALFIRE",
            severity    = sev,
            category    = "WILDFIRE",
            title       = f"{name} — {county_title} County",
            detail      = detail,
            location    = f"{county_title} County, CA",
            ts          = ts,
            distance_km = dist_km,
            lat         = lat,
            lon         = lon,
            url         = url_link or None,
            raw         = {"acres": acres, "contained": contained, "county": county},
        ))

    # Sort by severity then distance
    sev_order = {"RED": 0, "ORANGE": 1, "YELLOW": 2, "INFO": 3}
    items.sort(key=lambda x: (sev_order.get(x.severity, 9), x.distance_km or 9999))
    _cache_fire.update(items)
    return items


def get_fire_incidents() -> list[dict]:
    if _cache_fire.stale(TTL_FIRE):
        fetch_fire_incidents()
    return [i.to_dict() for i in _cache_fire.get()]


# ── PurpleAir API (AQI mesh) ─────────────────────────────────────
#
# Blitzortung (PLAN 2.10) and CA Power Outages (PLAN 2.12) were drafted in the
# 2026-05-19 overnight pass but shipped as non-functional stubs (dead REST URL
# / no usable free API). Stripped to keep this module honest. To re-attempt:
#   - Blitzortung: subscribe to MQTT `blitzortung.ha.sed.pl` via paho-mqtt
#     (already in requirements.txt) in a background thread.
#   - CA outages: scrape SDGE outage map or wait for a public REST tier.

_cache_aqi = _Cache()
TTL_AQI = 600  # 10 min — air quality is slow-moving

_PURPLEAIR_API_KEY = os.environ.get("PURPLEAIR_API_KEY", "")
_PURPLEAIR_URL = "https://api.purpleair.com/v1/sensors"
_last_aqi_result: dict = {"status": "uninitialized", "ts": 0}

def fetch_aqi_summary() -> dict:
    """
    Fetch PurpleAir AQI sensors within ~5km bounding box of home.
    Free tier: requires free API key from https://develop.purpleair.com/

    Returns: {median_aqi, max_aqi, sensor_count, ts, status}
    """
    if not _PURPLEAIR_API_KEY:
        return {
            "error": "PURPLEAIR_API_KEY not set",
            "median_aqi": None,
            "max_aqi": None,
            "sensor_count": 0,
            "status": "unconfigured",
            "ts": time.time(),
        }

    # Bounding box: ~5km = ~0.045 degrees lat/lon
    bbox_deg = 0.045
    params = (
        f"?api_key={_PURPLEAIR_API_KEY}"
        f"&nwlat={HOME_LAT + bbox_deg}&nwlng={HOME_LON - bbox_deg}"
        f"&selat={HOME_LAT - bbox_deg}&selng={HOME_LON + bbox_deg}"
        f"&fields=name,latitude,longitude,pm2_5_60minute"
    )
    url = _PURPLEAIR_URL + params
    data = _fetch(url, timeout=10)

    if not data or isinstance(data, dict) and data.get("error"):
        return {
            "error": str(data.get("error", "API error")) if isinstance(data, dict) else "API unreachable",
            "median_aqi": None,
            "max_aqi": None,
            "sensor_count": 0,
            "status": "error",
            "ts": time.time(),
        }

    sensors = data.get("data", []) if isinstance(data, dict) else []
    if not sensors:
        return {
            "median_aqi": None,
            "max_aqi": None,
            "sensor_count": 0,
            "status": "no sensors in zone",
            "ts": time.time(),
        }

    # Extract AQI values (PurpleAir PM2.5 → rough AQI conversion)
    # US EPA AQI: 0-50 (Good), 51-100 (Moderate), 101-150 (USG), 151-200 (Unhealthy), 200+ (Very Unhealthy)
    pm25_values = []
    for s in sensors:
        if isinstance(s, list) and len(s) > 3:
            pm25 = s[3]  # pm2_5_60minute field
            if pm25 is not None and isinstance(pm25, (int, float)):
                pm25_values.append(float(pm25))

    if not pm25_values:
        return {
            "median_aqi": None,
            "max_aqi": None,
            "sensor_count": len(sensors),
            "status": "no pm2.5 data",
            "ts": time.time(),
        }

    # Simple PM2.5 to AQI estimate (EPA formula simplified)
    def pm25_to_aqi(pm25):
        if pm25 <= 12:
            return pm25 * 50 / 12
        elif pm25 <= 35.4:
            return 50 + (pm25 - 12) * 50 / 23.4
        elif pm25 <= 55.4:
            return 100 + (pm25 - 35.4) * 50 / 20
        elif pm25 <= 150.4:
            return 150 + (pm25 - 55.4) * 50 / 95
        elif pm25 <= 250.4:
            return 200 + (pm25 - 150.4) * 50 / 100
        else:
            return 300 + (pm25 - 250.4) * 50 / 100

    aqi_values = [pm25_to_aqi(pm) for pm in pm25_values]
    aqi_values.sort()
    median_aqi = aqi_values[len(aqi_values) // 2]
    max_aqi = max(aqi_values)

    return {
        "median_aqi": round(median_aqi, 1),
        "max_aqi": round(max_aqi, 1),
        "sensor_count": len(sensors),
        "status": "ok",
        "ts": time.time(),
    }


def get_aqi_summary() -> dict:
    """Cached AQI summary. Re-fetches at most every TTL_AQI seconds."""
    global _last_aqi_result
    if _cache_aqi.stale(TTL_AQI):
        _last_aqi_result = fetch_aqi_summary()
        _cache_aqi.update([])  # touch cache so stale() resets the timer
    return _last_aqi_result


# ── Citizen App — Hyperlocal 911 Incidents ────────────────────────

_CITIZEN_TRENDING = (
    "https://citizen.com/api/incident/trending"
    f"?lowerLatitude={HOME_LAT - CRIME_RADIUS_DEG}"
    f"&lowerLongitude={HOME_LON - CRIME_RADIUS_DEG}"
    f"&upperLatitude={HOME_LAT + CRIME_RADIUS_DEG}"
    f"&upperLongitude={HOME_LON + CRIME_RADIUS_DEG}"
)
_CITIZEN_INC_URL = "https://citizen.com/api/incident/{key}"

_CITIZEN_HEADERS = {
    "User-Agent": "citizen/25 CFNetwork/1568.200.51 Darwin/24.1.0",
    "Accept": "application/json",
    "x-client-version": "6.0.0",
}

_CITIZEN_CATEGORY_MAP = {
    "shooting":     ("SHOOTING",    "RED"),
    "stabbing":     ("ASSAULT",     "RED"),
    "robbery":      ("ROBBERY",     "RED"),
    "assault":      ("ASSAULT",     "ORANGE"),
    "carjacking":   ("CARJACKING",  "RED"),
    "pursuit":      ("PURSUIT",     "ORANGE"),
    "fire":         ("FIRE",        "ORANGE"),
    "structure fire":("FIRE",       "RED"),
    "vehicle fire": ("FIRE",        "ORANGE"),
    "brush fire":   ("FIRE",        "ORANGE"),
    "ems":          ("EMS",         "YELLOW"),
    "crash":        ("ACCIDENT",    "YELLOW"),
    "traffic":      ("ACCIDENT",    "YELLOW"),
    "power outage": ("UTILITY",     "YELLOW"),
    "dui":          ("DUI",         "YELLOW"),
    "burglary":     ("BURGLARY",    "ORANGE"),
    "theft":        ("THEFT",       "YELLOW"),
    "suspicious":   ("SUSPICIOUS",  "YELLOW"),
    "search":       ("SEARCH",      "YELLOW"),
    "missing":      ("MISSING",     "ORANGE"),
    "homicide":     ("HOMICIDE",    "RED"),
    "overdose":     ("EMS",         "YELLOW"),
    "hazmat":       ("HAZMAT",      "ORANGE"),
}

_citizen_fetch_lock = threading.Lock()


def _classify_citizen(title: str, categories: list) -> tuple[str, str]:
    t = (title or "").lower()
    cats = [c.lower() for c in (categories or [])]
    all_text = t + " " + " ".join(cats)
    for keyword, (cat, sev) in _CITIZEN_CATEGORY_MAP.items():
        if keyword in all_text:
            return cat, sev
    return "INCIDENT", "INFO"


def fetch_citizen_incidents(max_fetch: int = 12) -> list[FeedItem]:
    """
    Pull trending Citizen incidents from the area.
    Fetches the trending index, then individual incident records.
    Rate-limited to avoid hammering the API.
    """
    with _citizen_fetch_lock:
        trending = _fetch(_CITIZEN_TRENDING, timeout=8, headers=_CITIZEN_HEADERS)
        if not trending:
            return []

        incident_ids = trending.get("results", [])
        if not incident_ids:
            return []

        items: list[FeedItem] = []
        now_ts = time.time()
        cutoff = now_ts - 86400  # only last 24 hours

        for key in incident_ids[:max_fetch]:
            if not isinstance(key, str):
                continue
            url  = _CITIZEN_INC_URL.format(key=key)
            data = _fetch(url, timeout=6, headers=_CITIZEN_HEADERS)
            if not data:
                continue

            ts_ms  = data.get("ts") or data.get("cs") or 0
            ts     = (ts_ms / 1000) if ts_ms > 1e10 else ts_ms
            if ts and ts < cutoff:
                continue

            if data.get("closed"):
                continue

            title      = data.get("title") or data.get("raw") or "Incident"
            raw_text   = data.get("raw") or title
            location   = data.get("rawLocation") or data.get("location") or HOME_NAME
            nbhd       = data.get("neighborhood") or location
            lat        = data.get("latitude")
            lon        = data.get("longitude")
            sev_raw    = (data.get("severity") or "").lower()
            categories = data.get("categories") or []

            dist_km = _haversine_km(HOME_LAT, HOME_LON, lat, lon) if lat and lon else None

            cat, sev = _classify_citizen(title, categories)

            # Override severity from Citizen's own field
            if sev_raw == "red":
                sev = "RED"
            elif sev_raw == "orange" and sev not in ("RED",):
                sev = "ORANGE"
            elif sev_raw == "yellow" and sev == "INFO":
                sev = "YELLOW"

            # Latest update text
            updates = data.get("updates") or {}
            update_texts = [v.get("text", "") for v in updates.values()
                            if v.get("type") not in ("ROOT",)]
            detail = update_texts[-1] if update_texts else raw_text
            detail = (detail or raw_text)[:300]

            items.append(FeedItem(
                source      = "CITIZEN",
                severity    = sev,
                category    = cat,
                title       = title,
                detail      = detail,
                location    = nbhd,
                ts          = ts or now_ts,
                distance_km = dist_km,
                lat         = lat,
                lon         = lon,
                url         = f"https://citizen.com/incident/{key}",
                raw         = {"key": key, "categories": categories, "sev_raw": sev_raw},
            ))
            time.sleep(0.15)  # gentle rate limiting

        # Sort: most severe first, then most recent
        sev_order = {"RED": 0, "ORANGE": 1, "YELLOW": 2, "INFO": 3}
        items.sort(key=lambda x: (sev_order.get(x.severity, 9), -(x.ts or 0)))
        _cache_citizen.update(items)
        return items


def get_citizen_incidents() -> list[dict]:
    if _cache_citizen.stale(TTL_CITIZEN):
        fetch_citizen_incidents()
    return [i.to_dict() for i in _cache_citizen.get()]


# ── Blitzortung lightning (MQTT, real-time) ─────────────────────────
#
# Subscribes to the community Blitzortung MQTT proxy used by the popular HA
# integration (mrk-its/homeassistant-blitzortung). Each strike message is a
# JSON object with lat/lon/mds/time. We compute distance via haversine and
# drop anything outside LIGHTNING_RADIUS_KM. Strikes within the radius are
# kept in a rolling deque (1h window) and exposed via get_lightning_recent().
#
# Background thread starts on first call. If paho-mqtt is missing or the
# broker is unreachable, the function returns an empty list — graceful no-op.

from collections import deque
import threading as _th

LIGHTNING_RADIUS_KM = float(os.environ.get("LIGHTNING_RADIUS_KM", "50"))
_LIGHTNING_BROKER = os.environ.get("LIGHTNING_MQTT_HOST", "blitzortung.ha.sed.pl")
_LIGHTNING_PORT   = int(os.environ.get("LIGHTNING_MQTT_PORT", "1883"))
# Geohash precision controls subscription breadth. Precision 2 ("9q" for SoCal)
# covers ~1250km × 625km — plenty wide. We filter in code.
_LIGHTNING_GEOHASH_PRECISION = int(os.environ.get("LIGHTNING_GEOHASH_PRECISION", "2"))

_lightning_strikes: deque = deque(maxlen=500)
_lightning_lock     = _th.Lock()
_lightning_thread: Optional[_th.Thread] = None
_lightning_started  = False
_lightning_status: dict = {"connected": False, "last_message_ts": None, "subscribed_topic": None, "error": None}


def _geohash_encode(lat: float, lon: float, precision: int = 8) -> str:
    """Inline geohash encoder — base32, 5 bits/char. No external dep."""
    base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    lat_lo, lat_hi = -90.0, 90.0
    lon_lo, lon_hi = -180.0, 180.0
    bit, ch_bits, out = 0, 0, []
    even = True
    while len(out) < precision:
        if even:
            mid = (lon_lo + lon_hi) / 2
            if lon >= mid: ch_bits |= (1 << (4 - bit)); lon_lo = mid
            else: lon_hi = mid
        else:
            mid = (lat_lo + lat_hi) / 2
            if lat >= mid: ch_bits |= (1 << (4 - bit)); lat_lo = mid
            else: lat_hi = mid
        even = not even
        bit += 1
        if bit == 5:
            out.append(base32[ch_bits])
            bit, ch_bits = 0, 0
    return "".join(out)


def _start_lightning_mqtt() -> None:
    """Start the MQTT subscriber in a daemon thread. Idempotent."""
    global _lightning_started, _lightning_thread
    if _lightning_started:
        return
    _lightning_started = True

    try:
        import paho.mqtt.client as mqtt  # noqa: F401
    except ImportError as e:
        _lightning_status["error"] = f"paho-mqtt not installed: {e}"
        return

    def _run() -> None:
        import paho.mqtt.client as mqtt
        gh = _geohash_encode(HOME_LAT, HOME_LON, _LIGHTNING_GEOHASH_PRECISION)
        topic = f"blitzortung/1.1/{'/'.join(gh)}/#"
        _lightning_status["subscribed_topic"] = topic

        def on_connect(client, userdata, flags, rc, properties=None):
            if rc == 0:
                _lightning_status["connected"] = True
                _lightning_status["error"] = None
                client.subscribe(topic, qos=0)
            else:
                _lightning_status["error"] = f"connect rc={rc}"

        def on_disconnect(client, userdata, *args, **kwargs):
            _lightning_status["connected"] = False

        def on_message(client, userdata, msg):
            try:
                payload = json.loads(msg.payload.decode("utf-8", errors="ignore"))
                lat = float(payload.get("lat"))
                lon = float(payload.get("lon"))
            except (ValueError, TypeError, json.JSONDecodeError):
                return
            dist = _haversine_km(HOME_LAT, HOME_LON, lat, lon)
            if dist > LIGHTNING_RADIUS_KM:
                return
            strike = {
                "lat": lat,
                "lon": lon,
                "distance_km": round(dist, 1),
                "ts": payload.get("time", time.time() * 1e9) / 1e9 if payload.get("time", 0) > 1e15 else (payload.get("time") or time.time()),
                "mds": payload.get("mds"),
            }
            with _lightning_lock:
                _lightning_strikes.append(strike)
            _lightning_status["last_message_ts"] = time.time()

        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"palm-command-{int(time.time())}")
        except (AttributeError, TypeError):
            # paho-mqtt < 2.x doesn't have CallbackAPIVersion
            client = mqtt.Client(client_id=f"palm-command-{int(time.time())}")
        client.on_connect = on_connect
        client.on_disconnect = on_disconnect
        client.on_message = on_message

        try:
            client.connect(_LIGHTNING_BROKER, _LIGHTNING_PORT, keepalive=60)
            client.loop_forever()
        except Exception as e:
            _lightning_status["error"] = f"connect_failed: {e}"
            _lightning_status["connected"] = False

    _lightning_thread = _th.Thread(target=_run, name="blitzortung-mqtt", daemon=True)
    _lightning_thread.start()


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def get_lightning_recent(max_age_seconds: int = 3600) -> list[dict]:
    """Return strikes from the last hour, newest first. Starts MQTT on first call."""
    _start_lightning_mqtt()
    now = time.time()
    cutoff = now - max_age_seconds
    with _lightning_lock:
        recent = [s for s in _lightning_strikes if (s.get("ts") or 0) >= cutoff]
    recent.sort(key=lambda s: s.get("ts") or 0, reverse=True)
    return recent


def lightning_summary() -> dict:
    """One-line dashboard summary."""
    strikes = get_lightning_recent()
    return {
        "source": "Blitzortung",
        "strike_count": len(strikes),
        "nearest_distance_km": min((s["distance_km"] for s in strikes), default=None),
        "last_strike_ts": (strikes[0]["ts"] if strikes else None),
        "mqtt_status": dict(_lightning_status),
    }


# ── Combined Feed ─────────────────────────────────────────────────

_all_cache:   list[dict] = []
_all_cache_ts: float     = 0.0
_all_lock                = threading.Lock()


def get_all_feeds(force: bool = False) -> dict:
    """
    Returns all feeds combined with a threat-level summary.
    Cached for TTL_ALL seconds. Thread-safe.
    """
    global _all_cache, _all_cache_ts

    now = time.time()
    with _all_lock:
        if not force and (now - _all_cache_ts) < TTL_ALL:
            return _build_combined(_all_cache)

    # Refresh all caches in parallel
    results: dict[str, list] = {}
    errors:  dict[str, str]  = {}

    def _run(name, fn):
        try:
            results[name] = fn()
        except Exception as e:
            errors[name] = str(e)
            results[name] = []

    threads = [
        threading.Thread(target=_run, args=("weather",  fetch_weather_alerts),   daemon=True),
        threading.Thread(target=_run, args=("quakes",   fetch_earthquakes),       daemon=True),
        threading.Thread(target=_run, args=("fire",     fetch_fire_incidents),    daemon=True),
        threading.Thread(target=_run, args=("citizen",  fetch_citizen_incidents), daemon=True),
    ]
    for t in threads: t.start()
    for t in threads: t.join(timeout=20)

    # Merge all items into flat list sorted by severity + recency
    all_items: list[FeedItem] = []
    for key in ("weather", "quakes", "fire", "citizen"):
        cache_map = {
            "weather":  _cache_weather,
            "quakes":   _cache_quake,
            "fire":     _cache_fire,
            "citizen":  _cache_citizen,
        }
        all_items.extend(cache_map[key].get())

    sev_order = {"RED": 0, "ORANGE": 1, "YELLOW": 2, "INFO": 3, "GREEN": 4}
    all_items.sort(key=lambda x: (sev_order.get(x.severity, 9), -(x.ts or 0)))

    all_dicts = [i.to_dict() for i in all_items]
    with _all_lock:
        _all_cache    = all_dicts
        _all_cache_ts = time.time()

    return _build_combined(all_dicts)


def _build_combined(items: list[dict]) -> dict:
    # Compute threat level
    sev_counts = {"RED": 0, "ORANGE": 0, "YELLOW": 0, "INFO": 0}
    for i in items:
        sev = i.get("severity", "INFO")
        sev_counts[sev] = sev_counts.get(sev, 0) + 1

    if sev_counts["RED"] > 0:
        threat = "RED"
        threat_label = "CRITICAL THREAT LEVEL"
    elif sev_counts["ORANGE"] >= 2:
        threat = "ORANGE"
        threat_label = "ELEVATED THREAT LEVEL"
    elif sev_counts["ORANGE"] > 0 or sev_counts["YELLOW"] >= 3:
        threat = "YELLOW"
        threat_label = "MODERATE THREAT LEVEL"
    else:
        threat = "GREEN"
        threat_label = "NOMINAL CONDITIONS"

    # Nearest significant threat
    nearest = None
    for i in items:
        if i.get("severity") in ("RED", "ORANGE") and i.get("distance_km") is not None:
            if nearest is None or i["distance_km"] < nearest["distance_km"]:
                nearest = i

    # Category breakdown
    cats: dict[str, int] = {}
    for i in items:
        c = i.get("category", "OTHER")
        cats[c] = cats.get(c, 0) + 1

    return {
        "threat_level":  threat,
        "threat_label":  threat_label,
        "severity_counts": sev_counts,
        "total":         len(items),
        "category_breakdown": cats,
        "nearest_threat": nearest,
        "items":         items,
        "last_updated":  datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "location":      HOME_NAME,
    }


# ── Background auto-refresh thread ───────────────────────────────

_bg_thread: threading.Thread | None = None
_bg_stop   = threading.Event()


def start_background_refresh(interval_s: int = 180):
    """
    Starts a background thread that refreshes all feeds every interval_s seconds.
    Call once at startup from camera_watcher.py.
    """
    global _bg_thread
    if _bg_thread and _bg_thread.is_alive():
        return

    def _loop():
        print(f"[feeds] Background refresh started (interval={interval_s}s)", flush=True)
        while not _bg_stop.wait(interval_s):
            try:
                get_all_feeds(force=True)
                print(f"[feeds] Refreshed: {len(_all_cache)} items", flush=True)
            except Exception as e:
                print(f"[feeds] Refresh error: {e}", flush=True)

    _bg_thread = threading.Thread(target=_loop, daemon=True, name="feeds-refresh")
    _bg_thread.start()


def stop_background_refresh():
    _bg_stop.set()


# ── Plain-English Summary (for PALANTIR) ─────────────────────────

def generate_briefing() -> str:
    """
    Returns a mission-briefing style text summary of all current feeds.
    Used by the PALANTIR query agent.
    """
    data = get_all_feeds()
    lines = [
        f"▸ AREA THREAT LEVEL: {data['threat_level']} — {data['threat_label']}",
        f"▸ LOCATION: {data['location']}",
        f"▸ {data['total']} active intelligence items · {data['last_updated']}",
        "",
    ]

    items = data.get("items", [])

    # Group by source
    by_source: dict[str, list] = {}
    for i in items:
        by_source.setdefault(i["source"], []).append(i)

    if by_source.get("USGS"):
        quakes = by_source["USGS"]
        lines.append(f"▸ SEISMIC — {len(quakes)} earthquakes detected in region:")
        for q in quakes[:4]:
            lines.append(f"  · {q['title']} — {q['detail']}")
        if len(quakes) > 4:
            lines.append(f"  · +{len(quakes)-4} smaller events")
        lines.append("")

    if by_source.get("NWS"):
        alerts = by_source["NWS"]
        lines.append(f"▸ WEATHER/NWS — {len(alerts)} active alert(s):")
        for a in alerts[:3]:
            lines.append(f"  · [{a['severity']}] {a['title']}")
            lines.append(f"    {a['detail'][:120]}")
        lines.append("")

    if by_source.get("CALFIRE"):
        fires = by_source["CALFIRE"]
        lines.append(f"▸ WILDFIRES — {len(fires)} active incident(s):")
        for f in fires[:4]:
            dist = f"  {f['distance_km']:.0f}km away · " if f.get("distance_km") else ""
            lines.append(f"  · [{f['severity']}] {f['title']}")
            lines.append(f"    {dist}{f['detail'][:120]}")
        lines.append("")

    if by_source.get("CITIZEN"):
        incidents = by_source["CITIZEN"]
        red_orange = [i for i in incidents if i["severity"] in ("RED", "ORANGE")]
        lines.append(f"▸ LOCAL INCIDENTS (Citizen) — {len(incidents)} total, {len(red_orange)} high-priority:")
        for i in incidents[:5]:
            dist = f"{i['distance_km']:.1f}km · " if i.get("distance_km") is not None else ""
            lines.append(f"  · [{i['severity']}] {i['title']}")
            lines.append(f"    {dist}{i['location']} — {i['detail'][:100]}")
        if len(incidents) > 5:
            lines.append(f"  · +{len(incidents)-5} additional incidents in area")
        lines.append("")

    if not items:
        lines.append("▸ NO ACTIVE INTELLIGENCE — All clear. Feeds nominal.")

    return "\n".join(lines)


# ── CLI test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Fetching intelligence feeds for {HOME_NAME} ({HOME_LAT}, {HOME_LON})")
    print("=" * 70)
    text = generate_briefing()
    print(text)
