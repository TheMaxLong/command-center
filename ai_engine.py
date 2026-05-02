#!/usr/bin/env python3.12
"""
YOLOv8 inference engine for PALM COMMAND.

Uses YOLOv8n (nano) for fast single-frame inference.
Filters to home-security-relevant COCO classes.

Device selection:
  - 'mps'  when running natively on Apple Silicon
  - 'cpu'  when running inside Docker on Mac (MPS not available in Linux VM)
  Auto-detected at startup.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional, Union

# Classes from COCO that matter for home security (id → label)
RELEVANT_CLASSES: dict[int, str] = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    24: "backpack",
    26: "handbag",
    28: "suitcase",
}

MIN_CONFIDENCE = float(os.environ.get("AI_MIN_CONF", "0.35"))
MODEL_SIZE     = os.environ.get("AI_MODEL", "yolov8n.pt")   # nano = fastest

_model = None
_device: str | None = None


def _get_device() -> str:
    """Pick best available inference device."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model():
    global _model, _device
    if _model is None:
        from ultralytics import YOLO
        _device = _get_device()
        print(f"[ai] Loading {MODEL_SIZE} on {_device}", flush=True)
        _model = YOLO(MODEL_SIZE)
    return _model


def detect(image_path: Union[str, Path]) -> list[dict]:
    """
    Run YOLOv8 detection on an image file.

    Returns a list of detections, each:
        {"class": str, "confidence": float, "bbox": [x1, y1, x2, y2]}

    Only RELEVANT_CLASSES above MIN_CONFIDENCE are returned.
    One result per class (highest confidence kept when duplicated).
    Returns [] on error or no relevant detections.
    """
    try:
        model   = _load_model()
        results = model.predict(
            str(image_path),
            device=_device,
            verbose=False,
            conf=MIN_CONFIDENCE,
        )
        raw: list[dict] = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in RELEVANT_CLASSES:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [round(float(v)) for v in box.xyxy[0]]
                raw.append({
                    "class":      RELEVANT_CLASSES[cls_id],
                    "confidence": round(conf, 3),
                    "bbox":       [x1, y1, x2, y2],
                })

        # Keep only highest-confidence detection per class
        best: dict[str, dict] = {}
        for d in raw:
            cls = d["class"]
            if cls not in best or d["confidence"] > best[cls]["confidence"]:
                best[cls] = d

        return list(best.values())

    except Exception as e:
        print(f"[ai] detect error: {e}", flush=True)
        return []


# ── Bounding-box annotation ───────────────────────────────────────

# Colours match the dashboard CSS variables
_BBOX_COLORS: dict[str, str] = {
    "person":     "#00d46a",
    "car":        "#00b8d9",
    "truck":      "#00b8d9",
    "bus":        "#00b8d9",
    "motorcycle": "#00b8d9",
    "bicycle":    "#00b8d9",
    "backpack":   "#f5c400",
    "handbag":    "#f5c400",
    "suitcase":   "#f5c400",
}


def annotate(
    image_path: Union[str, Path],
    detections: list[dict],
) -> Optional[Path]:
    """
    Draw bounding boxes + confidence labels on a copy of the snapshot.

    Saves  <same-dir>/snap_ann.jpg  (fixed name for easy serving).
    Returns the output path, or None on failure / no detections.
    """
    if not detections:
        return None
    try:
        from PIL import Image, ImageDraw

        p   = Path(image_path)
        img = Image.open(p).convert("RGB")
        drw = ImageDraw.Draw(img)

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            color = _BBOX_COLORS.get(d["class"], "#ffffff")

            # Bounding box (2 px border)
            drw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # Label pill above box
            label  = f"{d['class'].upper()} {int(d['confidence'] * 100)}%"
            pill_w = len(label) * 7 + 6
            pill_h = 14
            pill_y = max(0, y1 - pill_h)
            drw.rectangle([x1, pill_y, x1 + pill_w, pill_y + pill_h], fill=color)
            drw.text((x1 + 3, pill_y + 1), label, fill="#000000")

        out = p.parent / "snap_ann.jpg"
        img.save(str(out), "JPEG", quality=88)
        return out
    except Exception as e:
        print(f"[ai] annotate error: {e}", flush=True)
        return None


# ── Person crop extraction (for profiler) ────────────────────────

def extract_crops(
    image_path: Union[str, Path],
    detections: list[dict],
    cls_filter: str = "person",
) -> list[bytes]:
    """
    Return JPEG bytes for every bounding box matching *cls_filter*.
    Crops are resized to 64 × 128 px (standard person ReID input shape).
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
