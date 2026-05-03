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
        [{"class": str, "confidence": float, "bbox": [x1,y1,x2,y2], "instance": int}, ...]

    Unlike v1, multiple people in the same frame are all returned.
    """
    try:
        model   = _load_model()
        results = model.predict(
            str(image_path),
            device=_device,
            verbose=False,
            conf=MIN_CONFIDENCE,
        )

        detections: list[dict] = []
        class_counts: dict[str, int] = {}

        for r in results:
            # Sort by confidence desc so instance numbering is highest-conf first
            boxes = sorted(r.boxes, key=lambda b: float(b.conf[0]), reverse=True)
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id not in RELEVANT_CLASSES:
                    continue
                cls_name = RELEVANT_CLASSES[cls_id]
                conf     = float(box.conf[0])

                # Per-class threshold check
                if conf < CLASS_CONFIDENCE.get(cls_name, MIN_CONFIDENCE):
                    continue

                x1, y1, x2, y2 = [round(float(v)) for v in box.xyxy[0]]
                instance = class_counts.get(cls_name, 0)
                class_counts[cls_name] = instance + 1

                detections.append({
                    "class":      cls_name,
                    "confidence": round(conf, 3),
                    "bbox":       [x1, y1, x2, y2],
                    "instance":   instance,   # 0 = highest-conf of this class
                })

        return detections

    except Exception as e:
        print(f"[ai] detect error: {e}", flush=True)
        return []


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


def annotate(
    image_path: Union[str, Path],
    detections: list[dict],
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

            # Bounding box (2 px border + subtle corner ticks)
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
