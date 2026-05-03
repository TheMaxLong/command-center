#!/usr/bin/env python3.12
"""
PALM COMMAND — Neighborhood Traffic Camera Module

Provides a live-updating map view of San Rafael Ave & Palm Canyon Dr,
Palm Springs CA — stitched from OpenStreetMap tiles with tactical overlay.

No API keys required. OpenStreetMap tiles are free and public.
Refreshes every 10 minutes. Serves as a JPEG image.

Also supports any public MJPEG/JPEG camera URL via NEIGHBORHOOD_CAM_URL env var.

Endpoints exposed via camera_watcher.py:
  GET /trafficcam          → JPEG map view of the intersection
  GET /trafficcam/status   → JSON status of the traffic cam module
"""
from __future__ import annotations

import io
import math
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# ── Config ────────────────────────────────────────────────────────

# San Rafael Ave & Palm Canyon Dr, Palm Springs CA
TARGET_LAT  = float(os.environ.get("TRAFFIC_LAT",  "33.8240"))
TARGET_LON  = float(os.environ.get("TRAFFIC_LON", "-116.5424"))
TARGET_LABEL = os.environ.get("TRAFFIC_LABEL", "San Rafael Ave & Palm Canyon Dr")
TARGET_ZOOM = int(os.environ.get("TRAFFIC_ZOOM",   "17"))

# Optional: override with a direct camera URL (MJPEG or JPEG snapshot)
CAM_URL     = os.environ.get("NEIGHBORHOOD_CAM_URL", "")

CACHE_SEC   = int(os.environ.get("TRAFFIC_CACHE_SEC", "600"))   # 10 min
GRID        = int(os.environ.get("TRAFFIC_GRID", "3"))          # 3×3 tile grid
TILE_SIZE   = 256   # OSM tile pixels

_OSM_SERVERS = [
    "https://tile.openstreetmap.org",
    "https://a.tile.openstreetmap.org",
    "https://b.tile.openstreetmap.org",
    "https://c.tile.openstreetmap.org",
]

CACHE_PATH  = Path(os.environ.get("TRAFFIC_CACHE", "/tmp/trafficcam_cache.jpg"))

# ── Tile math ─────────────────────────────────────────────────────

def _latlon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Convert lat/lon to OSM tile (x, y) at a given zoom level."""
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    lat_r = math.radians(lat)
    y = int((1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n)
    return x, y


def _tile_to_latlon(x: int, y: int, zoom: int) -> tuple[float, float]:
    """Convert tile (x, y) at zoom to NW corner lat/lon."""
    n = 2 ** zoom
    lon = x / n * 360 - 180
    lat_r = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_r)
    return lat, lon


def _latlon_to_pixel(
    lat: float, lon: float,
    origin_tile_x: int, origin_tile_y: int,
    zoom: int,
    grid: int,
) -> tuple[int, int]:
    """Convert lat/lon to pixel position within the stitched map image."""
    n = 2 ** zoom
    # Fractional tile position
    tx = (lon + 180) / 360 * n
    lat_r = math.radians(lat)
    ty = (1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n

    # Pixel position relative to top-left of grid
    left_tile = origin_tile_x - grid // 2
    top_tile  = origin_tile_y - grid // 2
    px = int((tx - left_tile) * TILE_SIZE)
    py = int((ty - top_tile)  * TILE_SIZE)
    return px, py


# ── Tile fetcher ──────────────────────────────────────────────────

def _fetch_tile(zoom: int, x: int, y: int) -> Optional[bytes]:
    """Download a single OSM tile."""
    for server in _OSM_SERVERS:
        url = f"{server}/{zoom}/{x}/{y}.png"
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "PALM-COMMAND/2.0 (home-security; palm-springs-ca)",
                    "Accept": "image/png",
                },
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
                if data and len(data) > 100:
                    return data
        except Exception:
            continue
    return None


# ── Map stitcher ──────────────────────────────────────────────────

def _build_map() -> Optional[bytes]:
    """
    Fetch a GRID×GRID tile mosaic centered on the intersection and
    add a tactical crosshair + label overlay.
    Returns JPEG bytes or None on failure.
    """
    if not _PIL_OK:
        return None

    center_x, center_y = _latlon_to_tile(TARGET_LAT, TARGET_LON, TARGET_ZOOM)
    half  = GRID // 2
    total = GRID * TILE_SIZE

    canvas = Image.new("RGB", (total, total), color=(30, 40, 50))
    draw   = ImageDraw.Draw(canvas)

    tiles_loaded = 0
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            tx = center_x + dx
            ty = center_y + dy
            data = _fetch_tile(TARGET_ZOOM, tx, ty)
            if data:
                try:
                    tile_img = Image.open(io.BytesIO(data)).convert("RGB")
                    px = (dx + half) * TILE_SIZE
                    py = (dy + half) * TILE_SIZE
                    canvas.paste(tile_img, (px, py))
                    tiles_loaded += 1
                except Exception:
                    pass

    if tiles_loaded == 0:
        return None

    # ── Tactical dark overlay ──────────────────────────────────────
    # Darken the map slightly for tactical aesthetic
    overlay = Image.new("RGBA", canvas.size, (7, 11, 15, 100))
    canvas  = canvas.convert("RGBA")
    canvas  = Image.alpha_composite(canvas, overlay)
    canvas  = canvas.convert("RGB")
    draw    = ImageDraw.Draw(canvas)

    # Crosshair at intersection
    cx, cy = _latlon_to_pixel(TARGET_LAT, TARGET_LON, center_x, center_y, TARGET_ZOOM, GRID)

    # Outer ring
    r = 18
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(0, 212, 106), width=2)
    # Inner dot
    draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=(0, 212, 106))
    # Crosshair lines
    gap = 6
    draw.line([(cx-r-12, cy), (cx-gap, cy)], fill=(0, 212, 106), width=1)
    draw.line([(cx+gap, cy), (cx+r+12, cy)], fill=(0, 212, 106), width=1)
    draw.line([(cx, cy-r-12), (cx, cy-gap)], fill=(0, 212, 106), width=1)
    draw.line([(cx, cy+gap), (cx, cy+r+12)], fill=(0, 212, 106), width=1)

    # ── Labels ────────────────────────────────────────────────────
    try:
        font_big  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 13)
        font_med  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
        font_sm   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 10)
    except Exception:
        font_big = font_med = font_sm = ImageFont.load_default()

    # Top-left header bar
    header_h = 28
    draw.rectangle([0, 0, total, header_h], fill=(7, 11, 15, 220))
    draw.text((8, 6),  "▸ NEIGHBORHOOD OVERWATCH", fill=(0, 212, 106), font=font_big)
    draw.text((total - 160, 6), datetime.now().strftime("%H:%M:%S LOCAL"), fill=(61, 81, 102), font=font_med)

    # Intersection label
    lw = 220
    lh = 40
    draw.rectangle([cx - lw//2, cy + r + 4, cx + lw//2, cy + r + lh], fill=(7, 11, 15, 200))
    draw.text((cx - lw//2 + 6, cy + r + 8),  TARGET_LABEL, fill=(0, 212, 106), font=font_med)
    draw.text((cx - lw//2 + 6, cy + r + 22),
              f"{TARGET_LAT:.4f}°N  {abs(TARGET_LON):.4f}°W  ·  z{TARGET_ZOOM}",
              fill=(61, 81, 102), font=font_sm)

    # Bottom bar
    draw.rectangle([0, total - 20, total, total], fill=(7, 11, 15, 220))
    draw.text((8, total - 16), "© OpenStreetMap contributors", fill=(61, 81, 102), font=font_sm)
    draw.text((total - 148, total - 16), f"PALM COMMAND · D8", fill=(0, 184, 217), font=font_sm)

    # Tile count badge
    badge_txt = f"{tiles_loaded}/{GRID*GRID} tiles"
    draw.text((total - 82, 8), badge_txt, fill=(61, 81, 102), font=font_sm)

    # Grid lines (subtle)
    for i in range(GRID + 1):
        px = i * TILE_SIZE
        draw.line([(px, 0), (px, total)], fill=(15, 25, 35), width=1)
        draw.line([(0, px), (total, px)], fill=(15, 25, 35), width=1)

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=88, optimize=True)
    return buf.getvalue()


# ── Live camera URL fallback ──────────────────────────────────────

def _fetch_live_cam() -> Optional[bytes]:
    """Fetch a JPEG snapshot from a live camera URL if configured."""
    if not CAM_URL:
        return None
    try:
        req = urllib.request.Request(
            CAM_URL,
            headers={"User-Agent": "PALM-COMMAND/2.0"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            return resp.read()
    except Exception:
        return None


# ── Cache layer ───────────────────────────────────────────────────

_cache_data: Optional[bytes] = None
_cache_ts:   float = 0.0
_cache_lock  = threading.Lock()
_cache_tiles_loaded = 0
_last_status = "INIT"


def _refresh_cache() -> bool:
    global _cache_data, _cache_ts, _cache_tiles_loaded, _last_status

    # Try live cam first
    data = _fetch_live_cam()
    if data:
        with _cache_lock:
            _cache_data = data
            _cache_ts   = time.time()
            _last_status = "LIVE_CAM"
        return True

    # Build map
    data = _build_map()
    if data:
        CACHE_PATH.write_bytes(data)
        with _cache_lock:
            _cache_data = data
            _cache_ts   = time.time()
            _last_status = "OSM_MAP"
        print(f"[trafficcam] Map refreshed ({len(data)//1024}KB) — {TARGET_LABEL}", flush=True)
        return True

    _last_status = "FAILED"
    return False


def get_image() -> Optional[bytes]:
    """Return the current traffic cam image, refreshing if stale."""
    with _cache_lock:
        age = time.time() - _cache_ts
        data = _cache_data
        cached = data is not None and age < CACHE_SEC

    if cached:
        return data

    # Try to load from disk cache first while refreshing
    if CACHE_PATH.exists() and _cache_data is None:
        try:
            with _cache_lock:
                _cache_data = CACHE_PATH.read_bytes()
                _cache_ts   = CACHE_PATH.stat().st_mtime
        except Exception:
            pass

    # Refresh in background if not already refreshing
    threading.Thread(target=_refresh_cache, daemon=True,
                     name="trafficcam-refresh").start()

    with _cache_lock:
        return _cache_data


def get_status() -> dict:
    with _cache_lock:
        age  = int(time.time() - _cache_ts) if _cache_ts else None
        size = len(_cache_data) if _cache_data else 0
    return {
        "target":       TARGET_LABEL,
        "lat":          TARGET_LAT,
        "lon":          TARGET_LON,
        "zoom":         TARGET_ZOOM,
        "cache_age_s":  age,
        "cache_size_b": size,
        "cache_ttl_s":  CACHE_SEC,
        "status":       _last_status,
        "live_cam_url": CAM_URL or None,
        "pil_available":_PIL_OK,
        "grid":         f"{GRID}×{GRID}",
    }


# ── Background refresh thread ─────────────────────────────────────

def start_background_refresh():
    """Start a background thread that keeps the map fresh."""
    def _loop():
        # Initial fetch
        ok = _refresh_cache()
        print(f"[trafficcam] Initial fetch {'OK' if ok else 'FAILED'}", flush=True)
        while True:
            time.sleep(CACHE_SEC)
            try:
                _refresh_cache()
            except Exception as e:
                print(f"[trafficcam] Refresh error: {e}", flush=True)

    t = threading.Thread(target=_loop, daemon=True, name="trafficcam-bg")
    t.start()
    print(f"[trafficcam] Neighborhood overwatch: {TARGET_LABEL}", flush=True)
