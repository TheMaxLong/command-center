#!/usr/bin/env python3.12
"""
PALM COMMAND — optional vision intelligence helpers.

This module keeps higher-level scene geometry separate from camera capture:
zones, focus scoring, track summaries, and optional third-party CV tool status.
The live watcher can run without supervision/norfair installed; when they are
available, the status endpoint reports them as ready for future deeper use.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def _optional_import(name: str) -> bool:
    try:
        from importlib.util import find_spec
        return find_spec(name) is not None
    except Exception:
        return False


def capabilities() -> dict:
    """Report optional CV stack availability without making it a hard dependency."""
    return {
        "supervision": _optional_import("supervision"),
        "norfair": _optional_import("norfair"),
        "mode": "enhanced" if _optional_import("supervision") or _optional_import("norfair") else "core",
        "features": [
            "normalized_boxes",
            "pose_landmarks",
            "posture_tags",
            "attention_zone_scoring",
            "zone_occupancy",
            "track_summary",
            "operator_annotation_overlay",
        ],
    }


def image_size(image_path: Optional[str | Path]) -> tuple[int, int] | None:
    if not image_path:
        return None
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return None


def detection_center_norm(det: dict, frame_size: tuple[int, int] | None = None) -> tuple[float, float]:
    fw = float(det.get("frame_w") or (frame_size[0] if frame_size else 1) or 1)
    fh = float(det.get("frame_h") or (frame_size[1] if frame_size else 1) or 1)
    return float(det.get("cx", 0)) / fw, float(det.get("cy", 0)) / fh


def bbox_norm(det: dict, frame_size: tuple[int, int] | None = None) -> list[float]:
    fw = float(det.get("frame_w") or (frame_size[0] if frame_size else 1) or 1)
    fh = float(det.get("frame_h") or (frame_size[1] if frame_size else 1) or 1)
    x1, y1, x2, y2 = det.get("bbox") or [0, 0, 0, 0]
    return [
        round(max(0.0, min(1.0, float(x1) / fw)), 4),
        round(max(0.0, min(1.0, float(y1) / fh)), 4),
        round(max(0.0, min(1.0, float(x2) / fw)), 4),
        round(max(0.0, min(1.0, float(y2) / fh)), 4),
    ]


def _zone_contains(zone: dict, nx: float, ny: float) -> bool:
    return (
        float(zone.get("x1", 0.0)) <= nx <= float(zone.get("x2", 1.0))
        and float(zone.get("y1", 0.0)) <= ny <= float(zone.get("y2", 1.0))
    )


def enrich_detections(
    cam_id: str,
    detections: list[dict],
    snap_path: Optional[str | Path],
    attention_zones: list[dict] | None = None,
    known_zones: list[dict] | None = None,
) -> list[dict]:
    """Add normalized geometry, zone tags, and operator priority score."""
    if not detections:
        return detections
    size = image_size(snap_path)
    attn = attention_zones or []
    known = known_zones or []
    enriched: list[dict] = []

    for det in detections:
        d = dict(det)
        nx, ny = detection_center_norm(d, size)
        d["center_norm"] = [round(nx, 4), round(ny, 4)]
        d["bbox_norm"] = bbox_norm(d, size)

        focus_hits = [z for z in attn if _zone_contains(z, nx, ny)]
        known_hits = [z for z in known if _zone_contains(z, nx, ny)]
        if focus_hits:
            z = focus_hits[0]
            d["attention_zone"] = z.get("label", "focus")
            d["attention_priority"] = z.get("priority", "normal")
            d["operator_focus"] = True
        else:
            d["operator_focus"] = False
        if known_hits:
            d["known_zone"] = known_hits[0].get("label", "known")

        score = float(d.get("confidence") or 0.0)
        if d.get("class") == "person":
            score += 0.20
        if d.get("operator_focus"):
            score += 0.35 if d.get("attention_priority") == "high" else 0.20
        if d.get("known_zone"):
            score -= 0.20
        d["operator_score"] = round(max(0.0, min(1.0, score)), 3)
        d["operator_hint"] = _operator_hint(cam_id, d)
        enriched.append(d)
    return enriched


def _operator_hint(cam_id: str, det: dict) -> str:
    cls = det.get("class", "object")
    zone = det.get("attention_zone")
    known = det.get("known_zone")
    direction = det.get("direction")
    bits = [str(cls).upper()]
    if zone:
        bits.append(f"focus:{zone}")
    if known:
        bits.append(f"known:{known}")
    if direction and direction != "UNKNOWN":
        bits.append(str(direction).lower())
    return " · ".join(bits)


def scene_metrics(
    cam_id: str,
    detections: list[dict],
    attention_zones: list[dict] | None = None,
) -> dict:
    """Compact scene summary for APIs and UI cards."""
    attn = attention_zones or []
    by_class: dict[str, int] = {}
    zone_occupancy = {str(z.get("label", "focus")): 0 for z in attn}
    focus_hits = []
    tracks = []
    poses = []

    for d in detections or []:
        cls = d.get("class", "object")
        by_class[cls] = by_class.get(cls, 0) + 1
        if d.get("attention_zone"):
            zone = str(d.get("attention_zone"))
            zone_occupancy[zone] = zone_occupancy.get(zone, 0) + 1
            focus_hits.append({
                "class": cls,
                "zone": zone,
                "priority": d.get("attention_priority", "normal"),
                "score": d.get("operator_score", d.get("confidence", 0)),
                "hint": d.get("operator_hint", ""),
            })
        if d.get("track_id"):
            tracks.append({
                "track_id": d.get("track_id"),
                "class": cls,
                "direction": d.get("direction", "UNKNOWN"),
                "dwell_s": d.get("dwell_s", 0),
                "zone": d.get("attention_zone"),
                "state": d.get("track_state"),
                "pose": d.get("pose_status"),
                "pose_tags": d.get("pose_tags", []),
            })
        if cls == "person" and d.get("pose"):
            poses.append({
                "track_id": d.get("track_id"),
                "status": d.get("pose_status", "unknown"),
                "tags": d.get("pose_tags", []),
                "quality": (d.get("pose") or {}).get("quality"),
                "visible_points": (d.get("pose") or {}).get("visible_points"),
            })

    top = sorted(detections or [], key=lambda x: x.get("operator_score", 0), reverse=True)
    return {
        "camera": cam_id,
        "detections": len(detections or []),
        "by_class": by_class,
        "attention_zones": len(attn),
        "zone_occupancy": zone_occupancy,
        "focus_hits": focus_hits,
        "tracks": tracks,
        "poses": poses,
        "top_priority": top[0].get("operator_hint") if top else "none",
        "stack": capabilities(),
    }
