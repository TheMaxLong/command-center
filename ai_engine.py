#!/usr/bin/env python3.12
"""
PALM COMMAND — AI detection engine v2.

Upgrades over v1:
  • All instances per class returned (multiple people in frame)
  • Per-class confidence thresholds (persons stricter than vehicles)
  • Expanded COCO class set: animals, luggage, delivery items
  • 64-dim enriched embedding: colour histogram + spatial stats + edge density
  • YOLOv8s with auto-fallback to yolov8n for low-memory systems

Device selection:
  'mps'  — Apple Silicon native
  'cuda' — NVIDIA GPU if available
  'cpu'  — universal fallback
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional, Union

# ── COCO class map — expanded for home security ───────────────────
RELEVANT_CLASSES: dict[int, str] = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    14: "bird",
    15: "cat",
    16: "dog",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    28: "suitcase",
    63: "laptop",
    67: "cell phone",
}

# Per-class confidence thresholds — persons need higher confidence to
# avoid false positives; large vehicles can be looser.
CLASS_CONFIDENCE: dict[str, float] = {
    "person":     0.42,
    "bicycle":    0.35,
    "car":        0.30,
    "motorcycle": 0.35,
    "bus":        0.30,
    "truck":      0.30,
    "bird":       0.40,
    "cat":        0.40,
    "dog":        0.40,
    "backpack":   0.40,
    "umbrella":   0.38,
    "handbag":    0.40,
    "suitcase":   0.38,
    "laptop":     0.45,
    "cell phone": 0.45,
}

# Global minimum — any class below this is always dropped
MIN_CONFIDENCE = float(os.environ.get("AI_MIN_CONF", "0.30"))

# Model preference: try yolov8s first (better accuracy), fall back to nano
MODEL_SIZE = os.environ.get("AI_MODEL", "yolov8s.pt")

_model = None
_device: str | None = None


def _get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _load_model():
    global _model, _device
    if _model is None:
        from ultralytics import YOLO
        _device = _get_device()
        try:
            print(f"[ai] Loading {MODEL_SIZE} on {_device}", flush=True)
            _model = YOLO(MODEL_SIZE)
        except Exception:
            fallback = "yolov8n.pt"
            print(f"[ai] {MODEL_SIZE} failed, falling back to {fallback}", flush=True)
            _model = YOLO(fallback)
    return _model


def detect(image_path: Union[str, Path]) -> list[dict]:
    """
    Run detection on an image file.

    Returns ALL instances above their per-class threshold:
        [{
            "class": str, "confidence": float,
            "bbox": [x1,y1,x2,y2], "instance": int,
            "cx": int, "cy": int,           # bbox center
            "area": int,                    # bbox pixel area
            "frame_w": int, "frame_h": int  # image dimensions
        }, ...]

    Unlike v1, multiple people in the same frame are all returned.
    """
    try:
        from PIL import Image as _PilImage
        model   = _load_model()
        results = model.predict(
            str(image_path),
            device=_device,
            verbose=False,
            conf=MIN_CONFIDENCE,
        )

        # Get image dimensions for spatial context
        try:
            with _PilImage.open(image_path) as _img:
                frame_w, frame_h = _img.size
        except Exception:
            frame_w, frame_h = 1920, 1080

        detections: list[dict] = []
        class_counts: dict[str, int] = {}

        for r in results:
            boxes = sorted(r.boxes, key=lambda b: float(b.conf[0]), reverse=True)
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id not in RELEVANT_CLASSES:
                    continue
                cls_name = RELEVANT_CLASSES[cls_id]
                conf     = float(box.conf[0])

                if conf < CLASS_CONFIDENCE.get(cls_name, MIN_CONFIDENCE):
                    continue

                x1, y1, x2, y2 = [round(float(v)) for v in box.xyxy[0]]
                instance = class_counts.get(cls_name, 0)
                class_counts[cls_name] = instance + 1

                cx   = (x1 + x2) // 2
                cy   = (y1 + y2) // 2
                area = (x2 - x1) * (y2 - y1)

                detections.append({
                    "class":      cls_name,
                    "confidence": round(conf, 3),
                    "bbox":       [x1, y1, x2, y2],
                    "instance":   instance,
                    "cx":         cx,
                    "cy":         cy,
                    "area":       area,
                    "frame_w":    frame_w,
                    "frame_h":    frame_h,
                })

        return detections

    except Exception as e:
        print(f"[ai] detect error: {e}", flush=True)
        return []


# ── Per-camera person tracker ─────────────────────────────────────

class PersonTracker:
    """
    Lightweight frame-to-frame person tracker using IoU matching.

    Maintains persistent track IDs across detections, computes:
      - direction: APPROACHING / DEPARTING / STATIONARY / UNKNOWN
      - dwell_s:   seconds the track has been visible
      - track_id:  stable integer ID for this physical person

    One PersonTracker instance per camera. Thread-safe via the
    camera_watcher module-level lock (not internal).
    """

    # IoU threshold to link detection across frames
    IOU_THRESH  = 0.25
    # Area growth to call something "approaching" (>15% larger = approaching)
    APPROACH_THRESH = 0.15
    # Tracks expire after this many seconds without a match
    EXPIRE_S    = 30.0

    def __init__(self, cam_id: str):
        self.cam_id = cam_id
        # active_tracks: {track_id: {"bbox", "area", "cx", "cy", "first_ts", "last_ts", "direction", "history_area"}}
        self._tracks: dict[int, dict] = {}
        self._next_id = 1

    @staticmethod
    def _iou(a: list[int], b: list[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        ua = (ax2-ax1)*(ay2-ay1)
        ub = (bx2-bx1)*(by2-by1)
        return inter / (ua + ub - inter)

    def update(self, detections: list[dict], ts: float) -> list[dict]:
        """
        Match new detections to existing tracks.
        Returns detections enriched with track_id, direction, dwell_s.
        """
        import time as _time

        # Expire old tracks
        expired = [tid for tid, t in self._tracks.items()
                   if ts - t["last_ts"] > self.EXPIRE_S]
        for tid in expired:
            del self._tracks[tid]

        # Match detections to tracks via greedy IoU
        matched:   dict[int, int] = {}  # detection_idx → track_id
        used_tids: set[int]       = set()

        person_dets = [(i, d) for i, d in enumerate(detections) if d["class"] == "person"]

        for i, d in person_dets:
            best_iou = self.IOU_THRESH
            best_tid = None
            for tid, t in self._tracks.items():
                if tid in used_tids:
                    continue
                iou = self._iou(d["bbox"], t["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid is not None:
                matched[i]         = best_tid
                used_tids.add(best_tid)

        # Update matched tracks; create new ones for unmatched
        enriched = list(detections)
        for i, d in person_dets:
            if i in matched:
                tid   = matched[i]
                track = self._tracks[tid]
                # Direction based on area change
                prev_area = track.get("area", d["area"])
                if prev_area > 0:
                    delta = (d["area"] - prev_area) / prev_area
                    if delta > self.APPROACH_THRESH:
                        direction = "APPROACHING"
                    elif delta < -self.APPROACH_THRESH:
                        direction = "DEPARTING"
                    else:
                        direction = "STATIONARY"
                else:
                    direction = "UNKNOWN"
                dwell_s = ts - track["first_ts"]
                # Update track state
                track.update({
                    "bbox":    d["bbox"],
                    "area":    d["area"],
                    "cx":      d["cx"],
                    "cy":      d["cy"],
                    "last_ts": ts,
                    "direction": direction,
                })
            else:
                # New track
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {
                    "bbox":      d["bbox"],
                    "area":      d["area"],
                    "cx":        d["cx"],
                    "cy":        d["cy"],
                    "first_ts":  ts,
                    "last_ts":   ts,
                    "direction": "UNKNOWN",
                }
                direction = "UNKNOWN"
                dwell_s   = 0.0

            enriched[i] = {
                **d,
                "track_id":  tid,
                "direction": direction,
                "dwell_s":   round(dwell_s, 1),
            }

        return enriched


# ── Global tracker registry (one per camera) ──────────────────────
_trackers: dict[str, PersonTracker] = {}


def get_tracker(cam_id: str) -> PersonTracker:
    if cam_id not in _trackers:
        _trackers[cam_id] = PersonTracker(cam_id)
    return _trackers[cam_id]


# ── Bounding-box annotation ───────────────────────────────────────

_BBOX_COLORS: dict[str, str] = {
    "person":     "#00d46a",
    "bicycle":    "#00b8d9",
    "car":        "#00b8d9",
    "truck":      "#00b8d9",
    "bus":        "#00b8d9",
    "motorcycle": "#00b8d9",
    "bird":       "#f5c400",
    "cat":        "#f5c400",
    "dog":        "#f5c400",
    "backpack":   "#f5a623",
    "umbrella":   "#f5a623",
    "handbag":    "#f5a623",
    "suitcase":   "#f5a623",
    "laptop":     "#b07cff",
    "cell phone": "#b07cff",
}

_POSE_EDGES = [
    ("nose", "l_eye"), ("nose", "r_eye"), ("l_eye", "l_ear"), ("r_eye", "r_ear"),
    ("l_shoulder", "r_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"), ("l_shoulder", "l_hip"),
    ("r_shoulder", "r_hip"), ("l_hip", "r_hip"), ("l_hip", "l_knee"),
    ("l_knee", "l_ankle"), ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
]


def annotate(
    image_path: Union[str, Path],
    detections: list[dict],
    attention_zones: list[dict] | None = None,
    known_zones: list[dict] | None = None,
) -> Optional[Path]:
    """
    Draw bounding boxes + confidence labels on a copy of the snapshot.
    Handles multiple instances per class with numbered labels.
    Saves <same-dir>/snap_ann.jpg.
    """
    if not detections:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont

        p   = Path(image_path)
        img = Image.open(p).convert("RGB")
        drw = ImageDraw.Draw(img)
        iw, ih = img.size

        def _zone_rect(zone: dict) -> list[int]:
            return [
                int(float(zone.get("x1", 0.0)) * iw),
                int(float(zone.get("y1", 0.0)) * ih),
                int(float(zone.get("x2", 1.0)) * iw),
                int(float(zone.get("y2", 1.0)) * ih),
            ]

        def _draw_zone(zone: dict, color: str, prefix: str) -> None:
            x1, y1, x2, y2 = _zone_rect(zone)
            label = f"{prefix} {str(zone.get('label', 'ZONE')).upper()}"[:48]
            # sparse operator overlay: thin frame, corner ticks, translucent label strip
            drw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            tick = max(14, min(iw, ih) // 45)
            for ax, ay, sx, sy in [
                (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
            ]:
                drw.line([ax, ay, ax + sx * tick, ay], fill=color, width=3)
                drw.line([ax, ay, ax, ay + sy * tick], fill=color, width=3)
            pill_w = len(label) * 7 + 10
            pill_y = max(0, y1 - 18)
            drw.rectangle([x1, pill_y, min(iw, x1 + pill_w), pill_y + 16], fill="#000000")
            drw.text((x1 + 5, pill_y + 3), label, fill=color)

        for zone in attention_zones or []:
            color = "#f5c400" if zone.get("priority") == "high" else "#00b8d9"
            _draw_zone(zone, color, "FOCUS")

        for zone in known_zones or []:
            _draw_zone(zone, "#6f7d88", "KNOWN")

        # Track count per class for multi-instance labels
        class_seen: dict[str, int] = {}

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cls   = d["class"]
            color = _BBOX_COLORS.get(cls, "#ffffff")
            conf  = int(d["confidence"] * 100)

            # Count instances for this class
            idx = class_seen.get(cls, 0)
            class_seen[cls] = idx + 1

            # Label: "PERSON 87%" or "PERSON·2 87%" for multiple
            if class_seen[cls] > 1 or (d.get("instance", 0) > 0):
                label = f"{cls.upper()}·{idx+1} {conf}%"
            else:
                label = f"{cls.upper()} {conf}%"
            if d.get("track_id"):
                label += f" T{d.get('track_id')}"
            if d.get("attention_zone"):
                label += f" · {str(d.get('attention_zone')).upper()[:14]}"
            if d.get("pose_status") and d.get("pose_status") != "unknown":
                label += f" · {str(d.get('pose_status')).upper()[:10]}"

            # Bounding box (2 px border + subtle corner ticks)
            if d.get("operator_focus"):
                color = "#f5c400" if d.get("attention_priority") == "high" else "#00d46a"

            landmarks = {
                str(point.get("name")): point
                for point in (d.get("pose_landmarks") or [])
                if float(point.get("confidence", 0)) >= 0.28
            }
            if landmarks:
                pose_color = "#00d46a" if (d.get("pose") or {}).get("quality", 0) >= 0.55 else "#f5c400"
                for a, b in _POSE_EDGES:
                    pa, pb = landmarks.get(a), landmarks.get(b)
                    if not pa or not pb:
                        continue
                    drw.line(
                        [int(pa["x"]), int(pa["y"]), int(pb["x"]), int(pb["y"])],
                        fill=pose_color,
                        width=2,
                    )
                for point in landmarks.values():
                    px, py = int(point["x"]), int(point["y"])
                    drw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=pose_color)

            drw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            tick = 8
            for tx, ty in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
                drw.rectangle([tx-1, ty-1, tx+1, ty+1], fill=color)

            # Label pill above box
            pill_w = len(label) * 7 + 8
            pill_h = 16
            pill_y = max(0, y1 - pill_h - 1)
            drw.rectangle([x1, pill_y, x1 + pill_w, pill_y + pill_h], fill=color)
            drw.text((x1 + 4, pill_y + 2), label, fill="#000000")

        out = p.parent / "snap_ann.jpg"
        img.save(str(out), "JPEG", quality=88)
        return out
    except Exception as e:
        print(f"[ai] annotate error: {e}", flush=True)
        return None


# ── Enriched embedding (64-dim) for person profiling ─────────────

def compute_embedding(jpeg_bytes: bytes) -> list[float]:
    """
    64-dim appearance embedding combining:
      - 48-dim normalised RGB colour histogram (16 bins × 3 channels)
      - 8-dim spatial colour split (upper/lower half histograms, 4 bins each)
      - 4-dim brightness/contrast/saturation/edge statistics
      - 4-dim HSV hue histogram (broad hue zones)

    More discriminative than the v1 48-dim histogram while staying
    fast (pure PIL, no extra model needed).
    """
    from PIL import Image, ImageFilter
    import math

    img   = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    w, h  = img.size
    total = w * h
    emb: list[float] = []

    # ── 48-dim: 16-bin RGB histogram (full image) ─────────────────
    for ch in img.split():
        raw = ch.histogram()
        for i in range(16):
            emb.append(sum(raw[i * 16:(i + 1) * 16]) / total)

    # ── 8-dim: upper/lower body colour split (4 bins each) ────────
    upper = img.crop((0, 0, w, h // 2))
    lower = img.crop((0, h // 2, w, h))
    for half in (upper, lower):
        ht = half.width * half.height
        r_hist = half.split()[0].histogram()
        for i in range(4):
            emb.append(sum(r_hist[i * 64:(i + 1) * 64]) / ht)

    # ── 4-dim: brightness, contrast, saturation, edge energy ──────
    gray    = img.convert("L")
    gray_px = list(gray.getdata())
    mean_b  = sum(gray_px) / len(gray_px) / 255.0
    var_b   = sum((p / 255.0 - mean_b) ** 2 for p in gray_px) / len(gray_px)
    contrast = math.sqrt(var_b)

    hsv     = img.convert("HSV") if hasattr(Image, "HSV") else None
    if hsv:
        s_px    = list(hsv.split()[1].getdata())
        sat_avg = sum(s_px) / len(s_px) / 255.0
    else:
        sat_avg = 0.0

    edges   = gray.filter(ImageFilter.FIND_EDGES)
    e_px    = list(edges.getdata())
    edge_en = sum(p for p in e_px) / (len(e_px) * 255.0)

    emb += [mean_b, contrast, sat_avg, edge_en]

    # ── 4-dim: broad HSV hue zones (warm, cool, neutral, dark) ────
    try:
        import colorsys
        rgb_px   = list(img.getdata())
        hue_bins = [0.0, 0.0, 0.0, 0.0]
        for r, g, b in rgb_px[::16]:   # sample every 16th pixel for speed
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            if v < 0.15:
                hue_bins[3] += 1        # dark
            elif s < 0.15:
                hue_bins[2] += 1        # neutral/grey
            elif h < 0.17 or h > 0.92:
                hue_bins[0] += 1        # warm (red/orange/yellow)
            else:
                hue_bins[1] += 1        # cool (green/blue/purple)
        n = max(sum(hue_bins), 1)
        emb += [x / n for x in hue_bins]
    except Exception:
        emb += [0.0, 0.0, 0.0, 0.0]

    # Normalise to unit vector
    mag = math.sqrt(sum(x * x for x in emb)) or 1.0
    return [x / mag for x in emb]


# ── Person crop extraction ────────────────────────────────────────

def extract_crops(
    image_path: Union[str, Path],
    detections: list[dict],
    cls_filter: str = "person",
) -> list[bytes]:
    """
    Return JPEG bytes for every bounding box matching cls_filter.
    Crops resized to 64×128 px (standard person ReID input).
    All instances returned (not just the first).
    """
    try:
        from PIL import Image

        img   = Image.open(image_path).convert("RGB")
        crops: list[bytes] = []
        for d in detections:
            if d["class"] != cls_filter:
                continue
            x1, y1, x2, y2 = d["bbox"]
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img.crop((x1, y1, x2, y2)).resize((64, 128), Image.LANCZOS)
            buf  = io.BytesIO()
            crop.save(buf, "JPEG", quality=85)
            crops.append(buf.getvalue())
        return crops
    except Exception as e:
        print(f"[ai] extract_crops error: {e}", flush=True)
        return []
