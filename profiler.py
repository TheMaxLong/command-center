#!/usr/bin/env python3.12
"""
COMMAND CENTER — Person profiler v2.

Changes from v1:
  • Uses 64-dim enriched embedding from ai_engine.compute_embedding()
    (colour histogram + spatial split + brightness/contrast/edge/hue stats)
  • Profile dedup/merge: when two profiles are very similar (>= MERGE_THRESHOLD),
    they are silently merged, keeping the older one as canonical
  • Custom name labels stored in DB and returned via get_profile_label()
  • "first_seen_today" and "returning_today" flags in match results
  • Cross-camera tracking: profile.cameras updated across all camera IDs
  • Richer profiles_summary() with visit velocity and last_cam

Environment variables
─────────────────────
  PROFILE_MATCH_THRESH   cosine similarity cut-off  (default 0.88)
  PROFILE_MERGE_THRESH   threshold to auto-merge two profiles (default 0.97)
  PROFILE_MIN_SIGHTINGS  sightings before "REGULAR" label (default 4)
"""
from __future__ import annotations

import io
import json
import math
import os
import threading
from datetime import datetime, timezone, date
from typing import Optional

import event_db
import ai_engine

MATCH_THRESHOLD = float(os.environ.get("PROFILE_MATCH_THRESH",  "0.88"))
MERGE_THRESHOLD = float(os.environ.get("PROFILE_MERGE_THRESH",  "0.97"))
MIN_SIGHTINGS   = int  (os.environ.get("PROFILE_MIN_SIGHTINGS", "4"))

EMBED_ALPHA = 0.10   # EMA weight for new observations (lower = more stable)
DAYS = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]

_lock = threading.Lock()


# ── Embedding helpers ─────────────────────────────────────────────

def _cosine_sim(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _rolling_avg(stored: list[float], new: list[float], alpha: float = EMBED_ALPHA) -> list[float]:
    """Exponential moving average — profile adapts slowly over time."""
    if len(stored) != len(new):
        return new
    return [(1.0 - alpha) * s + alpha * n for s, n in zip(stored, new)]


def _embed_crop(jpeg_bytes: bytes) -> list[float]:
    """Use ai_engine's enriched 64-dim embedding."""
    return ai_engine.compute_embedding(jpeg_bytes)


# ── Profile merge helper ──────────────────────────────────────────

def _try_merge_profiles(profiles: list[dict]) -> None:
    """
    Scan all profile pairs. If two profiles have cosine similarity >= MERGE_THRESHOLD
    they are likely the same person seen under different lighting/clothing.
    Merge the newer into the older: transfer sightings, update cameras, delete newer.
    """
    if len(profiles) < 2:
        return
    merged: set[int] = set()
    for i, p1 in enumerate(profiles):
        if p1["id"] in merged:
            continue
        try:
            emb1 = json.loads(p1["embedding"])
        except Exception:
            continue
        for p2 in profiles[i + 1:]:
            if p2["id"] in merged:
                continue
            try:
                emb2 = json.loads(p2["embedding"])
            except Exception:
                continue
            sim = _cosine_sim(emb1, emb2)
            if sim >= MERGE_THRESHOLD:
                # Keep the older profile (lower id / earlier first_seen)
                keep_id = min(p1["id"], p2["id"], key=lambda pid: next(
                    (p["first_seen"] for p in profiles if p["id"] == pid), 0
                ))
                drop_id = p2["id"] if keep_id == p1["id"] else p1["id"]
                _merge_into(keep_id, drop_id, profiles)
                merged.add(drop_id)
                print(
                    f"[profiler] merged PROFILE-{drop_id:03d} → PROFILE-{keep_id:03d} "
                    f"(sim={sim:.3f})",
                    flush=True,
                )
                break  # restart after merge to avoid stale data


def _merge_into(keep_id: int, drop_id: int, profiles: list[dict]) -> None:
    """Transfer sightings from drop_id to keep_id and delete drop_id."""
    drop = next((p for p in profiles if p["id"] == drop_id), None)
    keep = next((p for p in profiles if p["id"] == keep_id), None)
    if not drop or not keep:
        return

    drop_cams  = json.loads(drop["cameras"])
    keep_cams  = json.loads(keep["cameras"])
    merged_cams = list(set(keep_cams + drop_cams))

    try:
        emb_keep = json.loads(keep["embedding"])
        emb_drop = json.loads(drop["embedding"])
        merged_emb = _rolling_avg(emb_keep, emb_drop, alpha=0.3)
    except Exception:
        merged_emb = json.loads(keep["embedding"])

    event_db.merge_profiles(keep_id, drop_id, merged_cams, merged_emb)


# ── Main API ──────────────────────────────────────────────────────

def match_or_create(
    cam_id: str,
    event_ts: float,
    crops: list[bytes],
    event_id: Optional[int] = None,
) -> list[int]:
    """
    For each person crop: find matching profile or create a new one.
    Returns list of profile IDs — one per crop.
    Thread-safe via module-level lock.
    """
    if not crops:
        return []

    matched_ids: list[int] = []

    with _lock:
        profiles = event_db.get_all_profiles()

        # Periodically attempt profile merging (every 20 calls roughly)
        import random
        if random.random() < 0.05 and len(profiles) >= 2:
            _try_merge_profiles(profiles)
            profiles = event_db.get_all_profiles()

        for crop_bytes in crops:
            try:
                hist = _embed_crop(crop_bytes)
            except Exception as e:
                print(f"[profiler] embed error: {e}", flush=True)
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
                stored_embed = json.loads(
                    next(p["embedding"] for p in profiles if p["id"] == best_id)
                )
                new_embed = _rolling_avg(stored_embed, hist)
                event_db.update_profile_sighting(
                    best_id, event_ts, cam_id, new_embed, event_id
                )
                matched_ids.append(best_id)
                sightings = next(p["sightings"] for p in profiles if p["id"] == best_id) + 1
                label = _make_label(best_id, sightings, None)
                print(f"[profiler] {cam_id} → {label} (sim={best_sim:.3f})", flush=True)
            else:
                pid = event_db.create_profile(cam_id, event_ts, hist, crop_bytes)
                matched_ids.append(pid)
                print(
                    f"[profiler] {cam_id} → NEW PROFILE-{pid:03d} (best_sim={best_sim:.3f})",
                    flush=True,
                )
                profiles = event_db.get_all_profiles()

    return matched_ids


# ── Labels and habits ─────────────────────────────────────────────

def _make_label(profile_id: int, sightings: int, override: Optional[str]) -> str:
    if override:
        return override
    if sightings >= MIN_SIGHTINGS:
        return f"REGULAR-{profile_id:03d}"
    return f"UNKNOWN-{profile_id:03d}"


def get_profile_label(profile_id: int) -> str:
    p = event_db.get_profile(profile_id)
    if not p:
        return f"UNKNOWN-{profile_id:03d}"
    return _make_label(profile_id, p["sightings"], p.get("label"))


def set_profile_label(profile_id: int, label: str) -> bool:
    """Persist a human-readable custom name for a profile."""
    return event_db.set_profile_label(profile_id, label.strip()[:48])


def get_habits(profile_id: int) -> dict:
    """
    Analyse time-of-day and day-of-week patterns for one profile.
    Returns: peak_hours, window, peak_days, total_seen, visits_last_7d,
             first_seen_today, returning_today
    """
    sightings = event_db.get_profile_sightings(profile_id)
    if not sightings:
        return {}

    now_utc = datetime.now(tz=timezone.utc)
    today   = now_utc.date()

    hours = [datetime.fromtimestamp(s["ts"], tz=timezone.utc).hour for s in sightings]
    days  = [datetime.fromtimestamp(s["ts"], tz=timezone.utc).weekday() for s in sightings]
    dates = [datetime.fromtimestamp(s["ts"], tz=timezone.utc).date() for s in sightings]

    def top_n(vals: list, n: int = 3) -> list:
        counts: dict = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        return sorted(counts, key=lambda k: counts[k], reverse=True)[:n]

    peak_hours = top_n(hours, 3)
    peak_days  = [DAYS[d] for d in top_n(days, 3)]

    sorted_ph = sorted(peak_hours)
    if sorted_ph:
        window = f"{sorted_ph[0]:02d}:00"
        if len(sorted_ph) > 1:
            window += f"\u2013{sorted_ph[-1] + 1:02d}:00"
    else:
        window = ""

    visits_last_7d = sum(
        1 for d in dates
        if (today - d).days < 7
    )

    today_sightings = [s for s in sightings if datetime.fromtimestamp(s["ts"], tz=timezone.utc).date() == today]
    returning_today = len(today_sightings) > 1
    first_seen_today = len(today_sightings) == 1

    return {
        "peak_hours":      peak_hours,
        "window":          window,
        "peak_days":       peak_days,
        "total_seen":      len(sightings),
        "visits_last_7d":  visits_last_7d,
        "first_seen_today": first_seen_today,
        "returning_today": returning_today,
    }


def profiles_summary() -> list[dict]:
    """Full profile list for the /profiles API endpoint."""
    profiles = event_db.get_all_profiles()
    result: list[dict] = []
    for p in profiles:
        pid   = p["id"]
        label = _make_label(pid, p["sightings"], p.get("label"))
        habits = get_habits(pid)
        result.append({
            "id":              pid,
            "label":           label,
            "sightings":       p["sightings"],
            "first_seen":      p["first_seen"],
            "last_seen":       p["last_seen"],
            "cameras":         json.loads(p["cameras"]),
            "habits":          habits,
            "has_thumb":       bool(p["thumb"]),
            "is_regular":      p["sightings"] >= MIN_SIGHTINGS,
            "returning_today": habits.get("returning_today", False),
            "first_seen_today":habits.get("first_seen_today", False),
            "visits_last_7d":  habits.get("visits_last_7d", 0),
        })
    return sorted(result, key=lambda x: x["last_seen"], reverse=True)
