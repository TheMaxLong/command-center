#!/usr/bin/env python3.12
"""
PALM COMMAND — Person profiler.

Builds appearance-based profiles from person crop images.
Matches incoming crops against known profiles via colour-histogram
cosine similarity.  Tracks time-of-day patterns to surface habits.

How it works
────────────
1. For each detected "person" crop, compute a 48-dim RGB histogram
   (16 bins × 3 channels).  Fast, no extra model, no GPU needed.
2. Compare against every stored profile embedding using cosine similarity.
3. If similarity ≥ MATCH_THRESHOLD  →  existing profile; update with
   exponential moving average so the profile adapts to lighting/clothing.
4. If no match  →  new profile created with that crop as its thumbnail.
5. Profiles accumulate sightings timestamped by camera + event.
6. get_habits() analyses peak hours and days per profile, surfacing
   patterns like "REGULAR-001 usually appears MON-FRI 08:00-09:00".

Environment variables
─────────────────────
  PROFILE_MATCH_THRESH   cosine similarity cut-off  (default 0.92)
  PROFILE_MIN_SIGHTINGS  sightings before "REGULAR" label (default 4)
"""
from __future__ import annotations

import io
import json
import math
import os
import threading
from datetime import datetime, timezone
from typing import Optional

import event_db

# ── Config ────────────────────────────────────────────────────────
MATCH_THRESHOLD = float(os.environ.get("PROFILE_MATCH_THRESH",  "0.92"))
MIN_SIGHTINGS   = int  (os.environ.get("PROFILE_MIN_SIGHTINGS", "4"))

# Weight given to each new observation in the rolling embedding average.
# Lower = slower to change (more stable long-term identity).
EMBED_ALPHA = 0.12

DAYS = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]

# Lock protects DB read-then-write inside match_or_create
_lock = threading.Lock()


# ── Appearance embedding ──────────────────────────────────────────

def _histogram(jpeg_bytes: bytes) -> list[float]:
    """
    Normalised 48-dim colour histogram (16 bins × RGB).

    Deliberately coarse — robust to minor pose/lighting changes while
    still distinguishing people by their dominant clothing colours.
    """
    from PIL import Image

    img   = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    total = img.width * img.height
    hist: list[float] = []
    for ch in img.split():           # R, G, B
        raw = ch.histogram()         # 256 values
        for i in range(16):
            hist.append(sum(raw[i * 16 : (i + 1) * 16]) / total)
    return hist


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _rolling_avg(stored: list[float], new: list[float]) -> list[float]:
    """Exponential moving average — profiles slowly adapt over time."""
    return [
        (1.0 - EMBED_ALPHA) * s + EMBED_ALPHA * n
        for s, n in zip(stored, new)
    ]


# ── Main API ──────────────────────────────────────────────────────

def match_or_create(
    cam_id: str,
    event_ts: float,
    crops: list[bytes],
    event_id: Optional[int] = None,
) -> list[int]:
    """
    For each person crop, find the best matching profile or create one.

    Returns a list of profile IDs — one per crop, in the same order.
    Thread-safe via a module-level lock.
    """
    if not crops:
        return []

    matched_ids: list[int] = []

    with _lock:
        profiles = event_db.get_all_profiles()

        for crop_bytes in crops:
            try:
                hist = _histogram(crop_bytes)
            except Exception as e:
                print(f"[profiler] histogram error: {e}", flush=True)
                continue

            best_id:  Optional[int] = None
            best_sim: float         = 0.0

            for p in profiles:
                try:
                    stored = json.loads(p["embedding"])
                    sim    = _cosine_sim(hist, stored)
                    if sim > best_sim:
                        best_sim = sim
                        best_id  = p["id"]
                except Exception:
                    continue

            if best_id is not None and best_sim >= MATCH_THRESHOLD:
                # Update rolling embedding for the matched profile
                stored_embed = json.loads(
                    next(p["embedding"] for p in profiles if p["id"] == best_id)
                )
                new_embed = _rolling_avg(stored_embed, hist)
                event_db.update_profile_sighting(
                    best_id, event_ts, cam_id, new_embed, event_id
                )
                matched_ids.append(best_id)
                sightings = next(
                    p["sightings"] for p in profiles if p["id"] == best_id
                ) + 1
                label = _make_label(best_id, sightings, None)
                print(
                    f"[profiler] {cam_id} → {label} (sim={best_sim:.3f})",
                    flush=True,
                )
            else:
                # New unknown person
                pid = event_db.create_profile(cam_id, event_ts, hist, crop_bytes)
                matched_ids.append(pid)
                print(
                    f"[profiler] {cam_id} → NEW PROFILE-{pid:03d} "
                    f"(best_sim={best_sim:.3f})",
                    flush=True,
                )
                # Refresh profile list for subsequent crops in same batch
                profiles = event_db.get_all_profiles()

    return matched_ids


# ── Label + habits ────────────────────────────────────────────────

def _make_label(profile_id: int, sightings: int, override: Optional[str]) -> str:
    if override:
        return override
    if sightings >= MIN_SIGHTINGS:
        return f"REGULAR-{profile_id:03d}"
    return f"UNKNOWN-{profile_id:03d}"


def get_profile_label(profile_id: int) -> str:
    """Human-readable label for a profile."""
    p = event_db.get_profile(profile_id)
    if not p:
        return f"UNKNOWN-{profile_id:03d}"
    return _make_label(profile_id, p["sightings"], p.get("label"))


def get_habits(profile_id: int) -> dict:
    """
    Analyse time-of-day and day-of-week patterns for one profile.

    Returns keys:
      peak_hours  – list of ints (0-23), up to 3 most common
      window      – human-readable time range string  e.g. "08:00–10:00"
      peak_days   – list of day abbreviations  e.g. ["MON", "TUE", "WED"]
      total_seen  – total sighting count
    """
    sightings = event_db.get_profile_sightings(profile_id)
    if not sightings:
        return {}

    hours = [
        datetime.fromtimestamp(s["ts"], tz=timezone.utc).hour
        for s in sightings
    ]
    days = [
        datetime.fromtimestamp(s["ts"], tz=timezone.utc).weekday()
        for s in sightings
    ]

    def top_n(vals: list[int], n: int = 3) -> list[int]:
        counts: dict[int, int] = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        return sorted(counts, key=lambda k: counts[k], reverse=True)[:n]

    peak_hours = top_n(hours, 3)
    peak_days  = [DAYS[d] for d in top_n(days, 3)]

    sorted_ph = sorted(peak_hours)
    if sorted_ph:
        window = f"{sorted_ph[0]:02d}:00"
        if len(sorted_ph) > 1:
            window += f"–{sorted_ph[-1] + 1:02d}:00"
    else:
        window = ""

    return {
        "peak_hours": peak_hours,
        "window":     window,
        "peak_days":  peak_days,
        "total_seen": len(sightings),
    }


def profiles_summary() -> list[dict]:
    """
    Full profile list for the /profiles API endpoint.
    Excludes raw embedding and thumbnail bytes.
    """
    profiles = event_db.get_all_profiles()
    result: list[dict] = []
    for p in profiles:
        pid   = p["id"]
        label = _make_label(pid, p["sightings"], p.get("label"))
        result.append({
            "id":         pid,
            "label":      label,
            "sightings":  p["sightings"],
            "first_seen": p["first_seen"],
            "last_seen":  p["last_seen"],
            "cameras":    json.loads(p["cameras"]),
            "habits":     get_habits(pid),
            "has_thumb":  bool(p["thumb"]),
            "is_regular": p["sightings"] >= MIN_SIGHTINGS,
        })
    return sorted(result, key=lambda x: x["last_seen"], reverse=True)
