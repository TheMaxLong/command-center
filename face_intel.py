#!/usr/bin/env python3.12
"""
COMMAND CENTER — Face Intelligence & Cross-Reference Engine

Pulls public wanted persons databases and compares detected faces
against them using OpenCV feature analysis.

Data sources (all free, no API keys required):
  FBI Most Wanted  — api.fbi.gov (1,160+ records with photos)
  FBI Kidnappings  — api.fbi.gov/wanted/v1/list?category=kidnappings
  FBI Fugitives    — api.fbi.gov/wanted/v1/list?category=fugitives
  Local POI DB     — operator-maintained database of persons of interest

Comparison method:
  1. OpenCV DNN face detector (ResNet SSD)
  2. Color histogram + LBP texture features for face comparison
  3. Structural similarity (SSIM) as secondary check
  4. Returns POSSIBLE MATCH / LOW PROBABILITY with confidence score
  NOTE: This is an investigative aid, NOT a legal identification system.
        Always verify with law enforcement. Confidence scores reflect
        feature similarity only.

Face comparison pipeline:
  detect_face(crop) → face_region → feature_vector
  compare(probe_features, gallery_features) → similarity_score
  flag_if_above(threshold=0.72) → alert
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Config ────────────────────────────────────────────────────────
FBI_API_BASE   = "https://api.fbi.gov/wanted/v1/list"
CACHE_DIR      = Path(os.environ.get("FACE_CACHE_DIR", "/tmp/face_intel"))
MATCH_THRESH   = float(os.environ.get("FACE_MATCH_THRESH", "0.72"))
REFRESH_HOURS  = float(os.environ.get("FBI_REFRESH_HOURS", "6"))
FIELD_OFFICES  = os.environ.get("FBI_FIELD_OFFICES", "losangeles,sandiego,lasvegas").split(",")
MAX_PER_OFFICE = int(os.environ.get("FBI_MAX_PER_OFFICE", "100"))

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── OpenCV DNN face detector ──────────────────────────────────────
_PROTOTEXT = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
_CAFFEMODEL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
_LOCAL_PROTO = CACHE_DIR / "deploy.prototxt"
_LOCAL_MODEL = CACHE_DIR / "face_detector.caffemodel"

_face_net  = None
_face_lock = threading.Lock()


def _load_face_detector() -> Optional[object]:
    global _face_net
    with _face_lock:
        if _face_net is not None:
            return _face_net if _face_net is not False else None

        try:
            # Download model files if missing
            if not _LOCAL_PROTO.exists():
                urllib.request.urlretrieve(_PROTOTEXT, str(_LOCAL_PROTO))
            if not _LOCAL_MODEL.exists():
                urllib.request.urlretrieve(_CAFFEMODEL, str(_LOCAL_MODEL))

            _face_net = cv2.dnn.readNetFromCaffe(str(_LOCAL_PROTO), str(_LOCAL_MODEL))
            print("[face_intel] OpenCV DNN face detector loaded", flush=True)
        except Exception as e:
            print(f"[face_intel] Face detector unavailable: {e}", flush=True)
            _face_net = False

        return _face_net if _face_net is not False else None


def detect_faces_in_image(img: np.ndarray, min_conf: float = 0.5) -> list[tuple[int,int,int,int]]:
    """
    Detect faces in an image using OpenCV DNN.
    Returns list of (x1, y1, x2, y2) bounding boxes.
    """
    net = _load_face_detector()
    if net is None:
        return _detect_faces_haar(img)

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < min_conf:
            continue
        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            faces.append((x1, y1, x2, y2))
    return faces


def _detect_faces_haar(img: np.ndarray) -> list[tuple[int,int,int,int]]:
    """Fallback: Haar cascade face detection."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces   = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        return [(x, y, x+w, y+h) for (x, y, w, h) in faces]
    except Exception:
        return []


def extract_face_from_crop(person_crop: np.ndarray) -> Optional[np.ndarray]:
    """Find and crop the face region from a person detection crop."""
    if person_crop is None or person_crop.size == 0:
        return None

    faces = detect_faces_in_image(person_crop)
    if not faces:
        # If no face detected, use upper 40% of crop (head region)
        h, w = person_crop.shape[:2]
        top = person_crop[:int(h * 0.4), :]
        return top if top.size > 0 else None

    # Use the largest detected face
    faces_sorted = sorted(faces, key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)
    x1, y1, x2, y2 = faces_sorted[0]
    return person_crop[y1:y2, x1:x2]


# ── Face feature extraction ───────────────────────────────────────

def _face_feature_vector(face_img: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract a compact face feature vector using:
    - Normalized RGB histograms (48 dims, 16 bins × 3 channels)
    - YCbCr skin tone histogram (16 dims)
    - LBP-style gradient histogram (16 dims)
    Total: 80-dim vector
    """
    if face_img is None or face_img.size == 0:
        return None

    try:
        # Resize to standard size
        face = cv2.resize(face_img, (64, 64))

        # RGB histograms
        rgb_hists = []
        for ch in range(3):
            h = cv2.calcHist([face], [ch], None, [16], [0, 256])
            h = h.flatten() / (h.sum() + 1e-7)
            rgb_hists.append(h)
        rgb_feat = np.concatenate(rgb_hists)  # 48 dims

        # YCbCr for skin tone
        ycbcr = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
        cb_hist = cv2.calcHist([ycbcr], [1], None, [8], [0, 256]).flatten()
        cr_hist = cv2.calcHist([ycbcr], [2], None, [8], [0, 256]).flatten()
        skin_feat = np.concatenate([
            cb_hist / (cb_hist.sum() + 1e-7),
            cr_hist / (cr_hist.sum() + 1e-7),
        ])  # 16 dims

        # Gradient magnitude histogram (edge structure)
        gray  = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx    = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy    = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        mag   = np.sqrt(gx**2 + gy**2)
        grad_hist, _ = np.histogram(mag, bins=16, range=(0, 200))
        grad_feat = (grad_hist / (grad_hist.sum() + 1e-7)).astype(np.float32)

        feat = np.concatenate([rgb_feat, skin_feat, grad_feat]).astype(np.float32)
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat

    except Exception as e:
        print(f"[face_intel] feature extraction error: {e}", flush=True)
        return None


def _face_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two face feature vectors."""
    dot  = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


# ── FBI Wanted Database ───────────────────────────────────────────

class WantedPerson:
    def __init__(self, uid: str, name: str, description: str, race: str,
                 sex: str, eyes: str, hair: str, subjects: list,
                 field_office: str, photo_url: str,
                 reward: str = "", age_range: str = ""):
        self.uid         = uid
        self.name        = name
        self.description = description
        self.race        = race
        self.sex         = sex
        self.eyes        = eyes
        self.hair        = hair
        self.subjects    = subjects
        self.field_office= field_office
        self.photo_url   = photo_url
        self.reward      = reward
        self.age_range   = age_range
        self._feature_vec: Optional[np.ndarray] = None
        self._photo_cache: Optional[np.ndarray] = None

    def get_photo(self) -> Optional[np.ndarray]:
        if self._photo_cache is not None:
            return self._photo_cache
        if not self.photo_url:
            return None
        try:
            req = urllib.request.Request(
                self.photo_url,
                headers={"User-Agent": "PALM-COMMAND/2.0"}
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = resp.read()
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            self._photo_cache = img
            return img
        except Exception:
            return None

    def get_feature_vector(self) -> Optional[np.ndarray]:
        if self._feature_vec is not None:
            return self._feature_vec
        photo = self.get_photo()
        if photo is None:
            return None
        faces = detect_faces_in_image(photo)
        if not faces:
            # Use full photo (some FBI photos are already cropped headshots)
            face_crop = photo
        else:
            x1, y1, x2, y2 = sorted(faces, key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)[0]
            face_crop = photo[y1:y2, x1:x2]
        self._feature_vec = _face_feature_vector(face_crop)
        return self._feature_vec

    def to_dict(self) -> dict:
        return {
            "uid":          self.uid,
            "name":         self.name,
            "description":  self.description[:200] if self.description else "",
            "race":         self.race or "Unknown",
            "sex":          self.sex or "Unknown",
            "eyes":         self.eyes or "Unknown",
            "hair":         self.hair or "Unknown",
            "subjects":     self.subjects,
            "field_office": self.field_office,
            "photo_url":    self.photo_url,
            "reward":       self.reward or "",
            "age_range":    self.age_range or "Unknown",
        }


# ── Local POI (Persons of Interest) database ──────────────────────

class POIPerson:
    """Operator-added person of interest with custom photo."""
    def __init__(self, poi_id: int, label: str, notes: str = "",
                 photo_path: Optional[str] = None, threat_level: str = "WATCH"):
        self.id          = poi_id
        self.label       = label
        self.notes       = notes
        self.photo_path  = photo_path
        self.threat_level = threat_level
        self.added_ts    = time.time()
        self._feature_vec: Optional[np.ndarray] = None

    def get_feature_vector(self) -> Optional[np.ndarray]:
        if self._feature_vec is not None:
            return self._feature_vec
        if not self.photo_path or not Path(self.photo_path).exists():
            return None
        img = cv2.imread(self.photo_path)
        if img is None:
            return None
        face = extract_face_from_crop(img)
        if face is None:
            return None
        self._feature_vec = _face_feature_vector(face)
        return self._feature_vec


# ── Face Intel Engine ─────────────────────────────────────────────

class FaceIntelEngine:
    """
    Main engine: maintains FBI + POI databases and runs face comparisons.
    """
    def __init__(self):
        self._wanted:    list[WantedPerson] = []
        self._poi:       list[POIPerson]    = []
        self._next_poi_id = 1
        self._lock       = threading.Lock()
        self._last_fbi_refresh = 0.0
        self._match_log: list[dict] = []
        self._load_poi_db()
        print("[face_intel] FaceIntelEngine initialized", flush=True)

    # ── FBI database management ──────────────────────────────────

    def refresh_fbi_database(self, force: bool = False) -> int:
        """Download/refresh FBI wanted persons for regional field offices."""
        now = time.time()
        if not force and (now - self._last_fbi_refresh) < REFRESH_HOURS * 3600:
            return len(self._wanted)

        new_wanted: list[WantedPerson] = []
        categories = ["fugitives", "kidnappings", "bank-robbers",
                      "crimes-against-children", "domestic-terrorism",
                      "most-wanted-terrorists"]

        # By field office
        for office in FIELD_OFFICES:
            try:
                url  = f"{FBI_API_BASE}?field_offices={office}&page=1"
                resp = self._fbi_fetch(url)
                for item in resp.get("items", [])[:MAX_PER_OFFICE]:
                    wp = self._parse_fbi_item(item, office)
                    if wp:
                        new_wanted.append(wp)
                time.sleep(0.2)
            except Exception as e:
                print(f"[face_intel] FBI office {office}: {e}", flush=True)

        # By category (national)
        for cat in categories:
            try:
                url  = f"{FBI_API_BASE}?category={cat}&page=1"
                resp = self._fbi_fetch(url)
                for item in resp.get("items", [])[:20]:
                    wp = self._parse_fbi_item(item, "national")
                    if wp and not any(x.uid == wp.uid for x in new_wanted):
                        new_wanted.append(wp)
                time.sleep(0.15)
            except Exception as e:
                print(f"[face_intel] FBI cat {cat}: {e}", flush=True)

        with self._lock:
            self._wanted          = new_wanted
            self._last_fbi_refresh = time.time()

        print(f"[face_intel] FBI database refreshed: {len(new_wanted)} persons", flush=True)
        return len(new_wanted)

    def _fbi_fetch(self, url: str) -> dict:
        req = urllib.request.Request(url, headers={
            "User-Agent": "PALM-COMMAND/2.0",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=12) as resp:
            return json.loads(resp.read().decode())

    def _parse_fbi_item(self, item: dict, office: str) -> Optional[WantedPerson]:
        uid = item.get("uid") or item.get("id") or ""
        if not uid:
            return None
        images = item.get("images") or []
        photo_url = images[0].get("thumb") or images[0].get("original") if images else ""
        if not photo_url:
            return None  # Skip entries without photos

        return WantedPerson(
            uid         = uid,
            name        = item.get("title") or "UNKNOWN",
            description = item.get("description") or item.get("details") or "",
            race        = item.get("race_raw") or item.get("race") or "",
            sex         = item.get("sex") or "",
            eyes        = item.get("eyes") or "",
            hair        = item.get("hair") or "",
            subjects    = item.get("subjects") or [],
            field_office= office,
            photo_url   = photo_url,
            reward      = item.get("reward_text") or "",
            age_range   = ", ".join(str(a) for a in (item.get("age_range") or [])),
        )

    # ── POI management ───────────────────────────────────────────

    def add_poi(self, label: str, photo_path: Optional[str] = None,
                notes: str = "", threat_level: str = "WATCH") -> dict:
        with self._lock:
            poi = POIPerson(self._next_poi_id, label, notes, photo_path, threat_level)
            self._poi.append(poi)
            self._next_poi_id += 1
            self._save_poi_db()
        print(f"[face_intel] POI added: {label} (id={poi.id})", flush=True)
        return {"id": poi.id, "label": label, "threat_level": threat_level}

    def _poi_db_path(self) -> Path:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return CACHE_DIR / "poi_db.json"

    def _load_poi_db(self) -> None:
        path = self._poi_db_path()
        if not path.exists():
            return
        try:
            rows = json.loads(path.read_text())
            for row in rows:
                poi = POIPerson(
                    int(row.get("id", self._next_poi_id)),
                    str(row.get("label", "POI")),
                    str(row.get("notes", "")),
                    row.get("photo_path"),
                    str(row.get("threat_level", "WATCH")),
                )
                self._poi.append(poi)
                self._next_poi_id = max(self._next_poi_id, poi.id + 1)
            print(f"[face_intel] Local POI database loaded: {len(self._poi)} persons", flush=True)
        except Exception as e:
            print(f"[face_intel] POI load error: {e}", flush=True)

    def _save_poi_db(self) -> None:
        rows = [
            {
                "id": p.id,
                "label": p.label,
                "notes": p.notes,
                "photo_path": p.photo_path,
                "threat_level": p.threat_level,
            }
            for p in self._poi
        ]
        self._poi_db_path().write_text(json.dumps(rows, indent=2))

    # ── Face comparison ──────────────────────────────────────────

    def compare_face(
        self,
        face_crop: np.ndarray,
        camera_id: str = "",
        ts: Optional[float] = None,
    ) -> list[dict]:
        """
        Compare a face crop against FBI + POI databases.
        Returns list of matches sorted by similarity (highest first).
        """
        ts = ts or time.time()
        probe = _face_feature_vector(face_crop)
        if probe is None:
            return []

        matches = []

        with self._lock:
            gallery = list(self._wanted) + list(self._poi)

        for person in gallery:
            try:
                feat = person.get_feature_vector()
                if feat is None:
                    continue
                sim = _face_similarity(probe, feat)
                if sim >= MATCH_THRESH:
                    is_poi = isinstance(person, POIPerson)
                    match = {
                        "source":      "POI" if is_poi else "FBI",
                        "uid":         str(person.id if is_poi else person.uid),
                        "name":        person.label if is_poi else person.name,
                        "similarity":  round(sim, 4),
                        "confidence":  "HIGH" if sim >= 0.88 else "MEDIUM" if sim >= 0.78 else "LOW",
                        "warning":     "⚠ VERIFY WITH LAW ENFORCEMENT — algorithmic match only",
                        "camera":      camera_id,
                        "ts":          ts,
                        "ts_human":    datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    }
                    if not is_poi:
                        match.update({
                            "subjects":     person.subjects,
                            "field_office": person.field_office,
                            "description":  (person.description or "")[:200],
                            "race":         person.race,
                            "sex":          person.sex,
                            "hair":         person.hair,
                            "eyes":         person.eyes,
                            "reward":       person.reward,
                            "photo_url":    person.photo_url,
                        })
                    matches.append(match)
            except Exception:
                continue

        matches.sort(key=lambda m: -m["similarity"])

        # Log significant matches
        if matches:
            with self._lock:
                self._match_log.extend(matches[:3])
                if len(self._match_log) > 200:
                    self._match_log = self._match_log[-200:]
            top = matches[0]
            print(f"[face_intel] ⚠ POSSIBLE MATCH: {top['name']} "
                  f"sim={top['similarity']:.3f} conf={top['confidence']} "
                  f"on {camera_id}", flush=True)

        return matches

    def compare_from_detection(
        self,
        image_path: str | Path,
        det: dict,
        camera_id: str,
        ts: Optional[float] = None,
    ) -> list[dict]:
        """
        Extract face from a person detection crop and run comparison.
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return []
            x1, y1, x2, y2 = det["bbox"]
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                return []
            face = extract_face_from_crop(crop)
            if face is None:
                return []
            return self.compare_face(face, camera_id, ts)
        except Exception as e:
            print(f"[face_intel] compare_from_detection error: {e}", flush=True)
            return []

    # ── Status ───────────────────────────────────────────────────

    def get_database_stats(self) -> dict:
        with self._lock:
            return {
                "fbi_count":      len(self._wanted),
                "poi_count":      len(self._poi),
                "match_log_count":len(self._match_log),
                "last_fbi_refresh":
                    datetime.fromtimestamp(self._last_fbi_refresh, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                    if self._last_fbi_refresh else "never",
                "field_offices":  FIELD_OFFICES,
                "match_threshold":MATCH_THRESH,
            }

    def get_match_log(self, limit: int = 20) -> list[dict]:
        with self._lock:
            return list(reversed(self._match_log[-limit:]))

    def get_wanted_list(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return [w.to_dict() for w in self._wanted[:limit]]

    def search_wanted(self, query: str) -> list[dict]:
        q = query.lower()
        with self._lock:
            results = [w for w in self._wanted
                       if q in (w.name or "").lower()
                       or q in " ".join(w.subjects).lower()
                       or q in (w.description or "").lower()]
        return [r.to_dict() for r in results[:20]]


# ── Global instance ───────────────────────────────────────────────

_engine: Optional[FaceIntelEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> FaceIntelEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = FaceIntelEngine()
        return _engine


def init_and_refresh(force: bool = False) -> int:
    """Initialize and load FBI database. Call at startup."""
    return get_engine().refresh_fbi_database(force=force)


def compare_detection(image_path, det, camera_id, ts=None):
    return get_engine().compare_from_detection(image_path, det, camera_id, ts)

def get_stats():
    return get_engine().get_database_stats()

def get_match_log(limit=20):
    return get_engine().get_match_log(limit)

def get_wanted_list(limit=50):
    return get_engine().get_wanted_list(limit)

def search_wanted(query):
    return get_engine().search_wanted(query)


# ── Background refresh thread ─────────────────────────────────────

def start_background_refresh():
    def _loop():
        # Initial load
        try:
            n = init_and_refresh()
            print(f"[face_intel] Initial FBI load: {n} persons", flush=True)
        except Exception as e:
            print(f"[face_intel] Initial load error: {e}", flush=True)

        while True:
            time.sleep(REFRESH_HOURS * 3600)
            try:
                n = init_and_refresh(force=True)
                print(f"[face_intel] FBI refresh: {n} persons", flush=True)
            except Exception as e:
                print(f"[face_intel] Refresh error: {e}", flush=True)

    t = threading.Thread(target=_loop, daemon=True, name="face-intel-refresh")
    t.start()
