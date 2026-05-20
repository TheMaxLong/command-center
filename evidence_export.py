#!/usr/bin/env python3.12
"""
COMMAND CENTER — Evidence Package Exporter

Bundles everything the system knows about an entity or profile into a
self-contained ZIP archive suitable for record-keeping or law enforcement.

Contents of generated ZIP:
  manifest.json       — package metadata (generated at, entity ID, version)
  report.txt          — human-readable incident report
  timeline.json       — all sightings sorted chronologically
  entity_profile.json — entity resolution record (cross-modal fusion data)
  pattern_of_life.json— behavioural model (schedule, dwell, frequency)
  face_matches.json   — any FBI/face-intel match records
  gait_data.json      — gait biometric signature
  snapshots/          — annotated JPEG frames from each sighting camera
    cam_<id>_snap.jpg
    cam_<id>_snap_ann.jpg

Usage:
  from evidence_export import generate_package
  zip_bytes = generate_package(entity_id="E0012345678", hours=72)
  # or by profile id:
  zip_bytes = generate_package(profile_id=3, hours=48)

HTTP route (camera_watcher.py):
  GET /api/evidence/<id>?hours=72   → application/zip download
  GET /api/evidence/profile/<id>
"""
from __future__ import annotations

import io, json, os, time, zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import event_db

MEDIA_DIR = Path(os.environ.get("MEDIA_DIR", "/tmp/cams"))
VERSION   = "1.0"


# ── Helpers ───────────────────────────────────────────────────────

def _ts_str(ts: Optional[float]) -> str:
    if not ts:
        return "—"
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def _safe_json(obj) -> str:
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return json.dumps({"error": "serialization failed"})


def _load_entity(entity_id: str) -> dict:
    try:
        import entity_resolution
        r = entity_resolution.get_resolver()
        return r.entity_detail(entity_id) or {}
    except Exception:
        return {}


def _load_profile(profile_id: int) -> dict:
    try:
        return event_db.get_profile(profile_id) or {}
    except Exception:
        return {}


def _load_sightings_for_profile(profile_id: int, since: float) -> list[dict]:
    try:
        rows = event_db.get_profile_sightings(profile_id)
        return [r for r in rows if r.get("ts", 0) >= since]
    except Exception:
        return []


def _load_pol(profile_id: int) -> dict:
    try:
        import pattern_engine
        eng = pattern_engine.get_engine()
        pols = eng.get_all_pol()
        for p in pols:
            eid = p.get("entity_id") or p.get("id") or ""
            if str(eid) == str(profile_id):
                return p
        return {}
    except Exception:
        return {}


def _load_face_matches(profile_id: int, since: float) -> list[dict]:
    try:
        import face_intel
        log = face_intel.get_match_log(200)
        return [m for m in log if m.get("ts", 0) >= since]
    except Exception:
        return []


def _load_gait(profile_id: int) -> dict:
    try:
        import gait_engine
        sigs = gait_engine.get_all_signatures()
        for s in sigs:
            if str(s.get("track_id", "")) == str(profile_id):
                return s
        return {}
    except Exception:
        return {}


def _collect_cameras_from_sightings(sightings: list[dict]) -> list[str]:
    seen = []
    for s in sightings:
        cam = s.get("cam_id") or s.get("camera")
        if cam and cam not in seen:
            seen.append(cam)
    return seen


def _snap_paths_for_camera(cam_id: str) -> dict[str, Optional[Path]]:
    base     = MEDIA_DIR / cam_id
    snap     = base / "snap.jpg"
    snap_ann = base / "snap_ann.jpg"
    return {
        "snap":     snap     if snap.exists()     and snap.stat().st_size > 0     else None,
        "snap_ann": snap_ann if snap_ann.exists() and snap_ann.stat().st_size > 0 else None,
    }


# ── Report builder ────────────────────────────────────────────────

def _build_report(
    entity_id: str,
    label: str,
    profile: dict,
    entity: dict,
    sightings: list[dict],
    pol: dict,
    face_matches: list[dict],
    gait: dict,
    since_ts: float,
    generated_at: str,
) -> str:
    lines = [
        "=" * 72,
        "  COMMAND CENTER — EVIDENCE PACKAGE",
        "=" * 72,
        f"  Generated:   {generated_at}",
        f"  Subject ID:  {entity_id}",
        f"  Label:       {label}",
        f"  Period:      {_ts_str(since_ts)} → {generated_at}",
        "",
        "── IDENTITY SUMMARY " + "─" * 52,
    ]

    # Profile basics
    if profile:
        first = _ts_str(profile.get("first_seen"))
        last  = _ts_str(profile.get("last_seen"))
        lines += [
            f"  First observed:   {first}",
            f"  Last observed:    {last}",
            f"  Total sightings:  {profile.get('sightings', 0)}",
            f"  Cameras seen on:  {', '.join(json.loads(profile.get('cameras') or '[]'))}",
        ]

    # Entity resolution
    if entity:
        conf = entity.get("confidence", 0)
        modalities = entity.get("modalities", {})
        lines += [
            "",
            f"  Identity confidence:  {conf*100:.1f}%",
            f"  Fusion modalities:    {', '.join(f'{k}:{v:.2f}' for k,v in modalities.items() if v)}",
        ]

    # Pattern of life
    lines.append("")
    lines.append("── PATTERN OF LIFE " + "─" * 52)
    if pol:
        lines += [
            f"  Behaviour class:    {pol.get('behaviour_class', '—')}",
            f"  Avg dwell (min):    {pol.get('avg_dwell_min', '—')}",
            f"  Avg interval (min): {pol.get('avg_inter_arrival_min', '—')}",
            f"  Peak hours:         {pol.get('peak_hours', '—')}",
            f"  Peak days:          {pol.get('peak_days', '—')}",
        ]
    else:
        lines.append("  Insufficient data for pattern-of-life modelling.")

    # Sightings timeline
    lines.append("")
    lines.append("── SIGHTINGS TIMELINE " + "─" * 49)
    if sightings:
        for s in sorted(sightings, key=lambda x: x.get("ts", 0)):
            ts_s   = _ts_str(s.get("ts"))
            cam    = s.get("cam_id") or s.get("camera", "?")
            lines.append(f"  {ts_s}  [{cam}]")
    else:
        lines.append("  No sightings in this time window.")

    # Face intelligence
    lines.append("")
    lines.append("── FACE INTELLIGENCE " + "─" * 50)
    if face_matches:
        for m in face_matches[:10]:
            sim  = m.get("similarity", 0)
            name = m.get("name", "?")
            cam  = m.get("camera_id", "?")
            ts_s = _ts_str(m.get("ts"))
            lines.append(f"  {ts_s}  [{cam}]  {name}  similarity={sim:.3f}")
    else:
        lines.append("  No FBI/face-intel matches in this time window.")

    # Gait biometrics
    lines.append("")
    lines.append("── GAIT BIOMETRICS " + "─" * 52)
    if gait:
        vec  = gait.get("signature", [])
        samp = gait.get("samples", 0)
        lines += [
            f"  Samples accumulated:  {samp}",
            f"  Signature dims:       {len(vec)}",
            f"  Vector (first 6):     {[round(v, 4) for v in vec[:6]]}",
        ]
    else:
        lines.append("  No gait signature collected for this subject.")

    lines += [
        "",
        "=" * 72,
        "  END OF EVIDENCE PACKAGE",
        "=" * 72,
    ]
    return "\n".join(lines)


# ── Main package generator ────────────────────────────────────────

def generate_package(
    entity_id:  Optional[str] = None,
    profile_id: Optional[int] = None,
    hours:      float         = 72.0,
) -> bytes:
    """
    Generate an evidence ZIP archive.

    Provide either entity_id (from entity_resolution) or profile_id (from
    profiler / event_db). Returns raw ZIP bytes.
    """
    if not entity_id and profile_id is None:
        raise ValueError("Provide entity_id or profile_id")

    since_ts     = time.time() - hours * 3600
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Resolve IDs
    pid_int: Optional[int] = None
    eid_str: str           = entity_id or ""

    if profile_id is not None:
        pid_int = int(profile_id)
        if not eid_str:
            eid_str = f"profile-{pid_int}"
    elif entity_id:
        # Try to find underlying profile_id from entity aliases
        try:
            import entity_resolution
            det = entity_resolution.get_resolver().entity_detail(entity_id)
            if det:
                aliases = det.get("aliases", [])
                for a in aliases:
                    try:
                        pid_int = int(a.lstrip("profile-").split("-")[-1])
                        break
                    except Exception:
                        pass
        except Exception:
            pass

    label = eid_str
    profile: dict     = {}
    entity: dict      = {}
    sightings: list   = []

    if pid_int is not None:
        profile   = _load_profile(pid_int)
        sightings = _load_sightings_for_profile(pid_int, since_ts)
        if profile.get("label"):
            label = profile["label"]

    if eid_str and not eid_str.startswith("profile-"):
        entity = _load_entity(eid_str)

    pol          = _load_pol(pid_int or 0)
    face_matches = _load_face_matches(pid_int or 0, since_ts)
    gait         = _load_gait(pid_int or 0)

    cameras_seen = _collect_cameras_from_sightings(sightings)
    if not cameras_seen and profile:
        try:
            cameras_seen = json.loads(profile.get("cameras") or "[]")
        except Exception:
            pass

    # Build report text
    report_txt = _build_report(
        entity_id=eid_str, label=label, profile=profile, entity=entity,
        sightings=sightings, pol=pol, face_matches=face_matches, gait=gait,
        since_ts=since_ts, generated_at=generated_at,
    )

    # Assemble ZIP in memory
    buf = io.BytesIO()
    safe_id = eid_str.replace("/", "-").replace("\\", "-")[:40]
    prefix  = f"palm_evidence_{safe_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:

        # manifest.json
        manifest = {
            "version":      VERSION,
            "generated_at": generated_at,
            "entity_id":    eid_str,
            "profile_id":   pid_int,
            "label":        label,
            "hours_window": hours,
            "since":        _ts_str(since_ts),
            "cameras":      cameras_seen,
            "sighting_count": len(sightings),
        }
        zf.writestr(f"{prefix}/manifest.json", _safe_json(manifest))

        # report.txt
        zf.writestr(f"{prefix}/report.txt", report_txt)

        # timeline.json
        zf.writestr(f"{prefix}/timeline.json",
                    _safe_json(sorted(sightings, key=lambda x: x.get("ts", 0))))

        # entity_profile.json
        if entity:
            zf.writestr(f"{prefix}/entity_profile.json", _safe_json(entity))
        elif profile:
            export_profile = {k: v for k, v in profile.items() if k != "thumb"}
            zf.writestr(f"{prefix}/entity_profile.json", _safe_json(export_profile))

        # pattern_of_life.json
        if pol:
            zf.writestr(f"{prefix}/pattern_of_life.json", _safe_json(pol))

        # face_matches.json
        if face_matches:
            zf.writestr(f"{prefix}/face_matches.json", _safe_json(face_matches))

        # gait_data.json
        if gait:
            zf.writestr(f"{prefix}/gait_data.json", _safe_json(gait))

        # snapshots/ — one snap + annotated per camera we've seen them on
        for cam_id in cameras_seen[:8]:  # cap at 8 cameras per package
            paths = _snap_paths_for_camera(cam_id)
            for kind, p in paths.items():
                if p:
                    try:
                        zf.writestr(f"{prefix}/snapshots/cam_{cam_id}_{kind}.jpg",
                                    p.read_bytes())
                    except Exception:
                        pass

    return buf.getvalue()


def package_filename(entity_id: str) -> str:
    safe = entity_id.replace("/", "-").replace("\\", "-")[:40]
    return f"palm_evidence_{safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
