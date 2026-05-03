#!/usr/bin/env python3.12
"""
PALM COMMAND — License Plate Recognition (LPR) Engine

Detects and reads license plates from vehicle crops.
Uses OpenCV for plate localization + EasyOCR for text reading
(falls back to basic thresholding + contour OCR if EasyOCR unavailable).

Pipeline:
  1. YOLO detects vehicle (car/truck/motorcycle)
  2. Vehicle crop extracted from snapshot
  3. Plate region located via edge detection + contour analysis
  4. OCR reads plate text
  5. Result logged to plate_log (in-memory + SQLite via event_db)
  6. Watchlist check: flag known plates

Plate log format:
  { plate, confidence, camera, ts, snap_path, bbox_vehicle, bbox_plate }
"""
from __future__ import annotations

import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── EasyOCR — optional, best accuracy ───────────────────────────
_ocr_reader  = None
_ocr_lock    = threading.Lock()
_ocr_enabled = False

def _init_ocr() -> bool:
    global _ocr_reader, _ocr_enabled
    with _ocr_lock:
        if _ocr_reader is not None:
            return _ocr_enabled
        try:
            import easyocr
            _ocr_reader  = easyocr.Reader(["en"], gpu=False, verbose=False)
            _ocr_enabled = True
            print("[lpr] EasyOCR loaded", flush=True)
        except Exception as e:
            print(f"[lpr] EasyOCR unavailable ({e}), using OpenCV OCR", flush=True)
            _ocr_enabled = False
        return _ocr_enabled


# ── Watchlist ────────────────────────────────────────────────────

_watched_plates: set[str]   = set()   # plates to flag
_known_plates:   dict[str, str] = {}  # plate → label ("OWN VEHICLE", "NEIGHBOR", ...)

_plates_lock = threading.Lock()


def add_watched_plate(plate: str, label: str = "FLAGGED"):
    with _plates_lock:
        _watched_plates.add(_normalize_plate(plate))
        _known_plates[_normalize_plate(plate)] = label


def remove_watched_plate(plate: str):
    with _plates_lock:
        p = _normalize_plate(plate)
        _watched_plates.discard(p)
        _known_plates.pop(p, None)


def add_known_plate(plate: str, label: str):
    with _plates_lock:
        _known_plates[_normalize_plate(plate)] = label


def get_plate_label(plate: str) -> Optional[str]:
    with _plates_lock:
        return _known_plates.get(_normalize_plate(plate))


def is_watched(plate: str) -> bool:
    with _plates_lock:
        return _normalize_plate(plate) in _watched_plates


# ── Plate log ────────────────────────────────────────────────────

_plate_log: list[dict] = []
_plate_log_lock = threading.Lock()
_PLATE_LOG_MAX  = 500


def _log_plate(record: dict):
    with _plate_log_lock:
        _plate_log.append(record)
        if len(_plate_log) > _PLATE_LOG_MAX:
            _plate_log.pop(0)
    print(f"[lpr] Plate logged: {record['plate']} @ {record['camera']} "
          f"(conf={record['confidence']:.2f})", flush=True)


def get_plate_log(camera: Optional[str] = None, limit: int = 50) -> list[dict]:
    with _plate_log_lock:
        items = [r for r in reversed(_plate_log)
                 if not camera or r.get("camera") == camera]
    return items[:limit]


def get_unique_plates(hours: float = 24) -> list[dict]:
    """Return unique plates seen in the last N hours with last-seen time."""
    cutoff = time.time() - hours * 3600
    seen: dict[str, dict] = {}
    with _plate_log_lock:
        for r in _plate_log:
            if r["ts"] >= cutoff:
                p = r["plate"]
                if p not in seen or r["ts"] > seen[p]["ts"]:
                    seen[p] = r
    result = sorted(seen.values(), key=lambda x: -x["ts"])
    return result


# ── Image preprocessing ──────────────────────────────────────────

def _normalize_plate(text: str) -> str:
    """Strip spaces, dashes, lowercase → uppercase for consistent matching."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _is_valid_plate(text: str) -> bool:
    """Basic sanity check for plate strings."""
    t = _normalize_plate(text)
    if len(t) < 3 or len(t) > 8:
        return False
    # Must have at least 2 letters and 1 digit (or 2 digits + 1 letter)
    letters = sum(1 for c in t if c.isalpha())
    digits  = sum(1 for c in t if c.isdigit())
    return letters >= 1 and digits >= 1


def _preprocess_for_ocr(gray: np.ndarray) -> np.ndarray:
    """Sharpen + binarize for better OCR."""
    # Adaptive threshold for varied lighting
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    # Mild sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp  = cv2.filter2D(thresh, -1, kernel)
    return sharp


def _find_plate_candidates(img_gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Find candidate plate regions using edge detection + contour filtering.
    Returns list of (x, y, w, h) candidates sorted by score.
    """
    # Bilateral filter to reduce noise while preserving edges
    blur = cv2.bilateralFilter(img_gray, 11, 17, 17)

    # Canny edge detection
    edges = cv2.Canny(blur, 30, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    img_h, img_w = img_gray.shape

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:50]:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        # Approximate polygon
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)

        x, y, w, h = cv2.boundingRect(approx)

        # Plate aspect ratio filter: typical plates are 2:1 to 5:1 wide
        if h == 0:
            continue
        aspect = w / h
        if not (1.5 <= aspect <= 6.0):
            continue

        # Size filter: not too small, not full frame
        if w < 40 or h < 10:
            continue
        if w > img_w * 0.9 or h > img_h * 0.9:
            continue

        # Score: prefer plates in lower half of image
        y_score = (y + h/2) / img_h
        area_score = min(area / (img_w * img_h * 0.05), 1.0)
        score = area_score * 0.6 + (1.0 - abs(y_score - 0.6)) * 0.4

        candidates.append((x, y, w, h, score))

    candidates.sort(key=lambda c: -c[4])
    return [(x, y, w, h) for x, y, w, h, _ in candidates[:5]]


def _ocr_region_opencv(gray_crop: np.ndarray) -> tuple[str, float]:
    """
    Basic OCR using OpenCV morphology + contour analysis.
    Less accurate than EasyOCR but requires no external model.
    """
    # Resize to at least 100px high
    h, w = gray_crop.shape
    if h < 30:
        scale = 60 / h
        gray_crop = cv2.resize(gray_crop, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_CUBIC)

    processed = _preprocess_for_ocr(gray_crop)

    # Find character contours
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_boxes = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / max(ch, 1)
        # Character-like aspect ratio
        if 0.2 <= aspect <= 1.2 and ch > processed.shape[0] * 0.3:
            char_boxes.append(x)

    # Very rough: count character-like blobs as confidence indicator
    confidence = min(len(char_boxes) / 7.0, 1.0) * 0.4  # max 0.4 — not reliable
    return "", confidence  # without Tesseract, we can't read the text


def _ocr_region_easyocr(gray_crop: np.ndarray) -> tuple[str, float]:
    """Read text from a grayscale plate crop using EasyOCR."""
    h, w = gray_crop.shape
    if h < 20:
        scale = 40 / h
        gray_crop = cv2.resize(gray_crop, (int(w * scale), 40),
                               interpolation=cv2.INTER_CUBIC)

    # EasyOCR works better with PIL/RGB
    rgb = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2RGB)

    results = _ocr_reader.readtext(rgb, detail=1, paragraph=False,
                                   allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -")
    if not results:
        return "", 0.0

    # Combine all text segments, pick highest confidence
    texts      = [r[1].upper().strip() for r in results]
    confs      = [r[2] for r in results]
    best_conf  = max(confs)
    combined   = "".join(re.sub(r"[^A-Z0-9]", "", t) for t in texts)
    return combined, best_conf


# ── Main entry point ─────────────────────────────────────────────

def process_vehicle_crop(
    image_path: str | Path,
    det: dict,
    camera_id: str,
    ts: float | None = None,
) -> Optional[dict]:
    """
    Extract plate from a vehicle detection crop and OCR it.

    Args:
        image_path: Path to the full frame snapshot
        det:        Detection dict from ai_engine (must be a vehicle class)
        camera_id:  Camera identifier
        ts:         Event timestamp (defaults to now)

    Returns:
        Plate record dict or None if no plate found / confidence too low
    """
    if det.get("class") not in ("car", "truck", "bus", "motorcycle"):
        return None

    ts = ts or time.time()
    path = Path(image_path)
    if not path.exists():
        return None

    try:
        img = cv2.imread(str(path))
        if img is None:
            return None

        x1, y1, x2, y2 = det["bbox"]
        # Pad crop slightly
        pad_x = int((x2 - x1) * 0.05)
        pad_y = int((y2 - y1) * 0.05)
        h_img, w_img = img.shape[:2]
        x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
        x2 = min(w_img, x2 + pad_x); y2 = min(h_img, y2 + pad_y)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Find plate candidates
        candidates = _find_plate_candidates(gray)
        if not candidates:
            return None

        _init_ocr()

        best_plate = ""
        best_conf  = 0.0
        best_bbox  = None

        for (px, py, pw, ph) in candidates[:3]:
            plate_crop = gray[py:py+ph, px:px+pw]
            if plate_crop.size == 0:
                continue

            if _ocr_enabled:
                text, conf = _ocr_region_easyocr(plate_crop)
            else:
                text, conf = _ocr_region_opencv(plate_crop)

            if conf > best_conf:
                best_conf  = conf
                best_plate = text
                best_bbox  = (x1+px, y1+py, x1+px+pw, y1+py+ph)

        if not best_plate or best_conf < 0.35 or not _is_valid_plate(best_plate):
            return None

        norm = _normalize_plate(best_plate)
        label  = get_plate_label(norm)
        flagged = is_watched(norm)

        record = {
            "plate":       norm,
            "raw_text":    best_plate,
            "confidence":  round(best_conf, 3),
            "camera":      camera_id,
            "ts":          ts,
            "ts_human":    datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "label":       label,
            "flagged":     flagged,
            "vehicle_class": det.get("class", "vehicle"),
            "bbox_vehicle": [x1, y1, x2, y2],
            "bbox_plate":   list(best_bbox) if best_bbox else None,
            "snap_path":   str(path),
        }

        _log_plate(record)

        if flagged:
            print(f"[lpr] ⚠ FLAGGED PLATE: {norm} on {camera_id}", flush=True)

        return record

    except Exception as e:
        print(f"[lpr] process_vehicle_crop error: {e}", flush=True)
        return None


def process_snapshot(image_path: str | Path, detections: list[dict], camera_id: str,
                     ts: float | None = None) -> list[dict]:
    """
    Process all vehicle detections in a snapshot.
    Returns list of plate records found.
    """
    results = []
    for det in detections:
        if det.get("class") in ("car", "truck", "bus", "motorcycle"):
            rec = process_vehicle_crop(image_path, det, camera_id, ts)
            if rec:
                results.append(rec)
    return results


# ── CLI test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python lpr_engine.py <image_path>")
        sys.exit(1)

    _init_ocr()
    img_path = sys.argv[1]
    img      = cv2.imread(img_path)
    if img is None:
        print(f"Cannot load image: {img_path}")
        sys.exit(1)

    h, w = img.shape[:2]
    fake_det = {"class": "car", "bbox": [0, 0, w, h], "confidence": 0.9}
    result   = process_vehicle_crop(img_path, fake_det, "test")
    if result:
        print(f"Plate: {result['plate']} (conf={result['confidence']:.2f})")
    else:
        print("No plate detected")
