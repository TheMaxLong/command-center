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

import os
from pathlib import Path
from typing import Union

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
