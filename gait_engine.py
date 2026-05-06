#!/usr/bin/env python3.12
"""
PALM COMMAND — Gait Analysis Engine

Identifies individuals by their walk signature — no face needed.
Uses YOLOv8-pose skeletal keypoints to compute a biometric gait vector
that persists across camera sessions, clothing changes, and lighting.

Technical basis:
  - YOLOv8n-pose: 17-keypoint COCO skeleton
  - Gait signature: 18-dimensional normalized biometric vector
  - Features: stride width, torso lean, hip sway, arm swing, head bob,
    shoulder-hip ratio, step symmetry, cadence index
  - Per-track temporal smoothing over last N frames
  - Cosine similarity matching against known gait profiles

Keypoint indices (COCO):
  0=nose  1=l_eye  2=r_eye  3=l_ear  4=r_ear
  5=l_shoulder  6=r_shoulder  7=l_elbow  8=r_elbow
  9=l_wrist  10=r_wrist  11=l_hip  12=r_hip
  13=l_knee  14=r_knee  15=l_ankle  16=r_ankle
"""
from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

# ── Model loading ─────────────────────────────────────────────────

_pose_model  = None
_pose_lock   = threading.Lock()
_POSE_MODEL  = os.environ.get("GAIT_MODEL", "yolov8n-pose.pt")
_POSE_CONF   = float(os.environ.get("GAIT_CONF", "0.4"))


def _load_pose_model():
    global _pose_model
    with _pose_lock:
        if _pose_model is None:
            try:
                from ultralytics import YOLO
                _pose_model = YOLO(_POSE_MODEL)
                print(f"[gait] Pose model loaded: {_POSE_MODEL}", flush=True)
            except Exception as e:
                print(f"[gait] Pose model load failed: {e}", flush=True)
                _pose_model = False
    return _pose_model or None


# ── Keypoint helpers ──────────────────────────────────────────────

_KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16,
}

_KP_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]


def _kp(kps: np.ndarray, name: str) -> Optional[tuple[float, float]]:
    """Get (x, y) of a named keypoint, or None if confidence too low."""
    idx = _KP.get(name)
    if idx is None or idx >= len(kps):
        return None
    x, y, conf = kps[idx]
    return (float(x), float(y)) if conf > 0.3 else None


def _dist(a: Optional[tuple], b: Optional[tuple]) -> Optional[float]:
    if a is None or b is None:
        return None
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def _midpoint(a: Optional[tuple], b: Optional[tuple]) -> Optional[tuple]:
    if a is None or b is None:
        return None
    return ((a[0]+b[0])/2, (a[1]+b[1])/2)


def _angle_deg(a: Optional[tuple], b: Optional[tuple],
               c: Optional[tuple]) -> Optional[float]:
    """Angle at vertex b, formed by a-b-c."""
    if any(p is None for p in (a, b, c)):
        return None
    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.sqrt(ba[0]**2+ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2+bc[1]**2)
    if mag_ba == 0 or mag_bc == 0:
        return None
    cos_a = max(-1, min(1, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_a))


def _pose_landmarks(kps: np.ndarray, frame_w: float, frame_h: float) -> list[dict]:
    """Compact serializable landmark set for API/UI overlays."""
    points: list[dict] = []
    fw = frame_w or 1.0
    fh = frame_h or 1.0
    for idx, name in enumerate(_KP_NAMES):
        if idx >= len(kps):
            continue
        x, y, conf = kps[idx]
        conf_f = float(conf)
        if conf_f < 0.20:
            continue
        points.append({
            "name": name,
            "x": round(float(x), 1),
            "y": round(float(y), 1),
            "nx": round(float(x) / fw, 4),
            "ny": round(float(y) / fh, 4),
            "confidence": round(conf_f, 3),
        })
    return points


def _pose_summary(kps: np.ndarray) -> dict:
    """Operator-facing pose tags. Heuristic, not biometric certainty."""
    nose       = _kp(kps, "nose")
    l_shoulder = _kp(kps, "l_shoulder")
    r_shoulder = _kp(kps, "r_shoulder")
    l_wrist    = _kp(kps, "l_wrist")
    r_wrist    = _kp(kps, "r_wrist")
    l_hip      = _kp(kps, "l_hip")
    r_hip      = _kp(kps, "r_hip")
    l_knee     = _kp(kps, "l_knee")
    r_knee     = _kp(kps, "r_knee")
    l_ankle    = _kp(kps, "l_ankle")
    r_ankle    = _kp(kps, "r_ankle")

    shoulder_mid = _midpoint(l_shoulder, r_shoulder)
    hip_mid      = _midpoint(l_hip, r_hip)
    knee_mid     = _midpoint(l_knee, r_knee)
    ankle_mid    = _midpoint(l_ankle, r_ankle)

    visible = sum(1 for i in range(min(len(kps), 17)) if float(kps[i][2]) >= 0.30)
    tags: list[str] = []
    posture = "unknown"

    if shoulder_mid and hip_mid:
        torso_dx = shoulder_mid[0] - hip_mid[0]
        torso_dy = abs(shoulder_mid[1] - hip_mid[1]) or 1.0
        lean = torso_dx / torso_dy
        if abs(lean) > 0.45:
            posture = "leaning"
            tags.append("torso lean")
        else:
            posture = "upright"

    if hip_mid and ankle_mid and shoulder_mid:
        body_h = max(abs(ankle_mid[1] - shoulder_mid[1]), 1.0)
        hip_drop = abs(ankle_mid[1] - hip_mid[1]) / body_h
        if hip_drop < 0.42:
            posture = "crouched"
            tags.append("low stance")

    if knee_mid and hip_mid and shoulder_mid:
        if knee_mid[1] < hip_mid[1] + abs(hip_mid[1] - shoulder_mid[1]) * 0.35:
            posture = "bending"
            tags.append("bending")

    arms_up = False
    if shoulder_mid:
        raised = []
        if l_wrist and l_wrist[1] < shoulder_mid[1]:
            raised.append("left")
        if r_wrist and r_wrist[1] < shoulder_mid[1]:
            raised.append("right")
        if raised:
            arms_up = True
            tags.append("hands raised" if len(raised) == 2 else f"{raised[0]} hand raised")

    if l_ankle and r_ankle and l_shoulder and r_shoulder:
        shoulder_w = _dist(l_shoulder, r_shoulder) or 1.0
        step_width = abs(l_ankle[0] - r_ankle[0]) / shoulder_w
        if step_width > 0.65:
            tags.append("wide stance")

    if not tags and posture != "unknown":
        tags.append(posture)

    return {
        "status": posture,
        "tags": tags[:4],
        "arms_up": arms_up,
        "visible_points": visible,
        "quality": round(visible / 17.0, 2),
    }


# ── Gait feature extraction ───────────────────────────────────────

def _compute_gait_features(kps: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute 18-dimensional gait feature vector from a single pose.

    All features are normalized to be scale-invariant (divided by body height
    or shoulder width so the vector doesn't depend on distance from camera).

    Returns None if too many keypoints are missing.
    """
    # Core landmarks
    nose       = _kp(kps, "nose")
    l_shoulder = _kp(kps, "l_shoulder")
    r_shoulder = _kp(kps, "r_shoulder")
    l_elbow    = _kp(kps, "l_elbow")
    r_elbow    = _kp(kps, "r_elbow")
    l_wrist    = _kp(kps, "l_wrist")
    r_wrist    = _kp(kps, "r_wrist")
    l_hip      = _kp(kps, "l_hip")
    r_hip      = _kp(kps, "r_hip")
    l_knee     = _kp(kps, "l_knee")
    r_knee     = _kp(kps, "r_knee")
    l_ankle    = _kp(kps, "l_ankle")
    r_ankle    = _kp(kps, "r_ankle")

    shoulder_mid = _midpoint(l_shoulder, r_shoulder)
    hip_mid      = _midpoint(l_hip, r_hip)

    # Require at minimum: shoulders, hips (for normalization)
    if l_shoulder is None or r_shoulder is None:
        return None
    if l_hip is None and r_hip is None:
        return None

    # Scale reference: shoulder width (pixels, unaffected by distance changes)
    shoulder_w = _dist(l_shoulder, r_shoulder)
    if not shoulder_w or shoulder_w < 5:
        return None

    # Body height estimate (nose-to-ankle or shoulder-to-ankle)
    body_h = None
    if nose and l_ankle:
        body_h = _dist(nose, l_ankle)
    elif shoulder_mid and l_ankle:
        body_h = _dist(shoulder_mid, l_ankle)
    if not body_h or body_h < 10:
        body_h = shoulder_w * 4  # fallback

    scale = shoulder_w  # normalize all distances by shoulder width

    feats = np.zeros(18, dtype=np.float32)

    # F0: Shoulder width / hip width ratio  (body shape)
    hip_w = _dist(l_hip, r_hip)
    feats[0] = (hip_w / shoulder_w) if hip_w else 0.85

    # F1: Torso length / shoulder width  (proportions)
    torso = _dist(shoulder_mid, hip_mid)
    feats[1] = (torso / scale) if torso else 0.0

    # F2: Torso lean angle from vertical (posture)
    if shoulder_mid and hip_mid:
        dx = shoulder_mid[0] - hip_mid[0]
        dy = shoulder_mid[1] - hip_mid[1]
        feats[2] = math.atan2(dx, abs(dy)) / math.pi  # -0.5..0.5
    else:
        feats[2] = 0.0

    # F3: Stride width — horizontal distance between ankles / shoulder_w
    ankle_dist_x = None
    if l_ankle and r_ankle:
        ankle_dist_x = abs(l_ankle[0] - r_ankle[0]) / scale
    feats[3] = ankle_dist_x if ankle_dist_x is not None else 0.0

    # F4: Step height — vertical ankle separation / shoulder_w (high stepping vs shuffle)
    ankle_dist_y = None
    if l_ankle and r_ankle:
        ankle_dist_y = abs(l_ankle[1] - r_ankle[1]) / scale
    feats[4] = ankle_dist_y if ankle_dist_y is not None else 0.0

    # F5: Left arm reach — l_wrist to l_shoulder / scale
    feats[5] = (_dist(l_wrist, l_shoulder) / scale) if (l_wrist and l_shoulder) else 0.0

    # F6: Right arm reach — r_wrist to r_shoulder / scale
    feats[6] = (_dist(r_wrist, r_shoulder) / scale) if (r_wrist and r_shoulder) else 0.0

    # F7: Arm swing asymmetry  |left_reach - right_reach|
    feats[7] = abs(feats[5] - feats[6])

    # F8: Left elbow angle  (arm bend)
    feats[8] = (_angle_deg(l_shoulder, l_elbow, l_wrist) or 180) / 180.0

    # F9: Right elbow angle
    feats[9] = (_angle_deg(r_shoulder, r_elbow, r_wrist) or 180) / 180.0

    # F10: Left knee angle
    feats[10] = (_angle_deg(l_hip, l_knee, l_ankle) or 180) / 180.0

    # F11: Right knee angle
    feats[11] = (_angle_deg(r_hip, r_knee, r_ankle) or 180) / 180.0

    # F12: Hip height relative to frame (low hip = bent-over posture)
    if hip_mid and l_ankle:
        feats[12] = _dist(hip_mid, l_ankle) / body_h
    else:
        feats[12] = 0.5

    # F13: Head height (nose y relative to shoulders)
    if nose and shoulder_mid:
        feats[13] = (_dist(nose, shoulder_mid) / scale) if True else 0.0
        feats[13] = min(feats[13], 2.0)

    # F14: Left wrist height relative to hip (hand position while walking)
    if l_wrist and hip_mid:
        feats[14] = (hip_mid[1] - l_wrist[1]) / scale  # positive = hands up
    else:
        feats[14] = 0.0

    # F15: Right wrist height relative to hip
    if r_wrist and hip_mid:
        feats[15] = (hip_mid[1] - r_wrist[1]) / scale
    else:
        feats[15] = 0.0

    # F16: Hip sway — hip midpoint x relative to shoulder midpoint x
    if hip_mid and shoulder_mid:
        feats[16] = (hip_mid[0] - shoulder_mid[0]) / scale
    else:
        feats[16] = 0.0

    # F17: Leg length asymmetry — |l_leg - r_leg| / scale
    l_leg = _dist(l_hip, l_ankle) if (l_hip and l_ankle) else None
    r_leg = _dist(r_hip, r_ankle) if (r_hip and r_ankle) else None
    if l_leg and r_leg:
        feats[17] = abs(l_leg - r_leg) / scale
    else:
        feats[17] = 0.0

    return feats


# ── Per-track gait accumulator ────────────────────────────────────

class GaitTrack:
    """
    Accumulates gait feature vectors across frames for a single track.
    Maintains a smoothed gait signature.
    """
    WINDOW = 20          # frames to average
    MIN_FRAMES = 5       # minimum frames before signature is "stable"

    def __init__(self, track_id: int):
        self.track_id = track_id
        self._frames: deque[np.ndarray] = deque(maxlen=self.WINDOW)
        self.first_ts = time.time()
        self.last_ts  = self.first_ts
        self.frame_count = 0

    def add_frame(self, feats: np.ndarray, ts: float):
        self._frames.append(feats)
        self.last_ts = ts
        self.frame_count += 1

    @property
    def stable(self) -> bool:
        return len(self._frames) >= self.MIN_FRAMES

    @property
    def signature(self) -> Optional[np.ndarray]:
        """Smoothed mean gait signature vector."""
        if not self._frames:
            return None
        arr = np.array(self._frames)
        mean = arr.mean(axis=0)
        # Normalize to unit vector
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 0 else mean

    @property
    def variance(self) -> Optional[float]:
        """Intra-track consistency (low = very consistent gait)."""
        if len(self._frames) < 3:
            return None
        arr = np.array(self._frames)
        return float(arr.std(axis=0).mean())


# ── Gait profile database ─────────────────────────────────────────

class GaitProfile:
    """A persistent gait identity — one per recognized individual."""
    def __init__(self, gait_id: int, signature: np.ndarray,
                 label: str = "", person_profile_id: Optional[int] = None):
        self.id                = gait_id
        self.signature         = signature
        self.label             = label
        self.person_profile_id = person_profile_id
        self.sightings         = 1
        self.first_seen        = time.time()
        self.last_seen         = time.time()
        self.cameras: list[str] = []

    def update(self, new_sig: np.ndarray, camera: str):
        """Update signature with exponential moving average."""
        alpha = 0.2  # weight of new observation
        updated = (1 - alpha) * self.signature + alpha * new_sig
        norm    = np.linalg.norm(updated)
        self.signature = updated / norm if norm > 0 else updated
        self.sightings += 1
        self.last_seen  = time.time()
        if camera not in self.cameras:
            self.cameras.append(camera)

    def similarity(self, other_sig: np.ndarray) -> float:
        """Cosine similarity [0, 1] with another signature."""
        dot  = np.dot(self.signature, other_sig)
        norm = np.linalg.norm(self.signature) * np.linalg.norm(other_sig)
        return float(dot / norm) if norm > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "id":                self.id,
            "label":             self.label or f"GAIT-{self.id:03d}",
            "sightings":         self.sightings,
            "first_seen":        self.first_seen,
            "last_seen":         self.last_seen,
            "cameras":           self.cameras,
            "person_profile_id": self.person_profile_id,
            "signature_norm":    float(np.linalg.norm(self.signature)),
        }


# ── Main Gait Engine ──────────────────────────────────────────────

class GaitEngine:
    MATCH_THRESHOLD  = 0.88   # cosine similarity — above = same person
    STRONG_THRESHOLD = 0.94   # high-confidence match
    MAX_TRACKS       = 50     # max in-memory track accumulators

    def __init__(self):
        self._tracks:   dict[int, GaitTrack]   = {}  # track_id → accumulator
        self._profiles: list[GaitProfile]       = []
        self._next_id   = 1
        self._lock      = threading.Lock()
        print("[gait] GaitEngine initialized", flush=True)

    def process_frame(
        self,
        image_path: str | Path,
        detections: list[dict],
        camera_id: str,
        ts: float | None = None,
    ) -> list[dict]:
        """
        Run pose estimation on snapshot, extract gait features for each
        tracked person, and update gait profiles.

        Returns detections enriched with 'gait_id', 'gait_label', 'gait_conf'.
        """
        model = _load_pose_model()
        if not model:
            return detections

        ts = ts or time.time()
        enriched = list(detections)

        try:
            results = model(str(image_path), conf=_POSE_CONF, verbose=False)
        except Exception as e:
            print(f"[gait] inference error: {e}", flush=True)
            return detections

        if not results:
            return detections

        result = results[0]
        if result.keypoints is None:
            return detections

        kps_all = result.keypoints.data.cpu().numpy()  # shape [N, 17, 3]
        frame_h, frame_w = result.orig_shape if getattr(result, "orig_shape", None) else (1, 1)

        # Match each pose to a tracked person detection by bbox IoU
        for det_idx, det in enumerate(detections):
            if det.get("class") != "person":
                continue

            track_id = det.get("track_id")
            if not track_id:
                continue

            det_bbox = det["bbox"]

            # Find best-matching pose by bbox overlap
            best_pose_idx = None
            best_iou      = 0.3
            for pose_idx, box in enumerate(result.boxes.xyxy.cpu().numpy()):
                bx1, by1, bx2, by2 = box
                # Quick IoU
                ix1 = max(det_bbox[0], bx1); iy1 = max(det_bbox[1], by1)
                ix2 = min(det_bbox[2], bx2); iy2 = min(det_bbox[3], by2)
                inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                if inter == 0: continue
                ua = ((det_bbox[2]-det_bbox[0])*(det_bbox[3]-det_bbox[1])
                      + (bx2-bx1)*(by2-by1) - inter)
                iou = inter / ua if ua > 0 else 0
                if iou > best_iou:
                    best_iou      = iou
                    best_pose_idx = pose_idx

            if best_pose_idx is None:
                continue

            kps   = kps_all[best_pose_idx]  # [17, 3]
            feats = _compute_gait_features(kps)
            pose_summary = _pose_summary(kps)
            pose_landmarks = _pose_landmarks(kps, float(frame_w), float(frame_h))

            # Update track accumulator
            gait_id = None
            gait_label = None
            gait_conf = 0.0
            gait_stable = False
            gait_frames = 0
            with self._lock:
                if track_id not in self._tracks:
                    if len(self._tracks) > self.MAX_TRACKS:
                        # Evict oldest
                        oldest = min(self._tracks, key=lambda k: self._tracks[k].last_ts)
                        del self._tracks[oldest]
                    self._tracks[track_id] = GaitTrack(track_id)

                gtrack = self._tracks[track_id]
                if feats is not None:
                    gtrack.add_frame(feats, ts)

                    if gtrack.stable:
                        sig   = gtrack.signature
                        match = self._match_profile(sig)
                        if match:
                            match.update(sig, camera_id)
                            gait_id    = match.id
                            gait_label = match.label or f"GAIT-{match.id:03d}"
                            gait_conf  = match.similarity(sig)
                        else:
                            # Create new gait profile
                            prof = GaitProfile(
                                gait_id   = self._next_id,
                                signature = sig,
                                person_profile_id = det.get("profile_id"),
                            )
                            prof.cameras.append(camera_id)
                            self._profiles.append(prof)
                            gait_id    = self._next_id
                            gait_label = f"GAIT-{self._next_id:03d}"
                            gait_conf  = 1.0
                            self._next_id += 1
                            print(f"[gait] New gait profile: {gait_label} on {camera_id}", flush=True)
                gait_stable = gtrack.stable
                gait_frames = gtrack.frame_count

            enriched[det_idx] = {
                **det,
                "pose": pose_summary,
                "pose_status": pose_summary.get("status"),
                "pose_tags": pose_summary.get("tags", []),
                "pose_landmarks": pose_landmarks,
                "gait_id":    gait_id,
                "gait_label": gait_label,
                "gait_conf":  round(gait_conf, 3),
                "gait_stable": gait_stable,
                "gait_frames": gait_frames,
            }

        return enriched

    def _match_profile(self, sig: np.ndarray) -> Optional[GaitProfile]:
        """Find the best-matching gait profile above threshold."""
        best_sim  = self.MATCH_THRESHOLD
        best_prof = None
        for prof in self._profiles:
            sim = prof.similarity(sig)
            if sim > best_sim:
                best_sim  = sim
                best_prof = prof
        return best_prof

    def link_gait_to_person(self, gait_id: int, person_profile_id: int,
                             label: str = ""):
        """Link a gait profile to a known person profile."""
        with self._lock:
            for prof in self._profiles:
                if prof.id == gait_id:
                    prof.person_profile_id = person_profile_id
                    if label:
                        prof.label = label
                    print(f"[gait] GAIT-{gait_id:03d} linked to person {person_profile_id}", flush=True)
                    return

    def get_profiles(self) -> list[dict]:
        with self._lock:
            return [p.to_dict() for p in self._profiles]

    def get_track_status(self, track_id: int) -> Optional[dict]:
        with self._lock:
            gt = self._tracks.get(track_id)
            if not gt:
                return None
            return {
                "track_id":    track_id,
                "frames":      gt.frame_count,
                "stable":      gt.stable,
                "variance":    round(gt.variance or 0, 4),
                "signature":   (gt.signature.tolist() if gt.signature is not None else None),
            }

    def analyze_gait_features(self, track_id: int) -> Optional[dict]:
        """Human-readable breakdown of a track's gait characteristics."""
        with self._lock:
            gt = self._tracks.get(track_id)
            if not gt or not gt.stable:
                return None
            sig = gt.signature
            if sig is None:
                return None

        # Map features to human descriptors
        desc = {}
        desc["shoulder_hip_ratio"]  = "wide shoulders" if sig[0] > 0.9 else "narrow shoulders"
        desc["torso_length"]        = "long torso" if sig[1] > 1.0 else "short torso"
        desc["posture"]             = (
            "leans right" if sig[2] > 0.05 else
            "leans left"  if sig[2] < -0.05 else
            "upright"
        )
        desc["stride_width"]        = "wide gait" if sig[3] > 0.5 else "narrow gait"
        desc["step_height"]         = "high stepper" if sig[4] > 0.3 else "flat-footed"
        desc["arm_swing"]           = "large arm swing" if max(sig[5], sig[6]) > 1.2 else "minimal arm swing"
        desc["arm_symmetry"]        = "symmetric" if sig[7] < 0.1 else "asymmetric arms"
        desc["elbow_bend_left"]     = f"L elbow {sig[8]*180:.0f}°"
        desc["elbow_bend_right"]    = f"R elbow {sig[9]*180:.0f}°"
        desc["knee_bend_left"]      = "bent knee L" if sig[10] < 0.8 else "straight knee L"
        desc["knee_bend_right"]     = "bent knee R" if sig[11] < 0.8 else "straight knee R"
        desc["hip_height"]          = "low center of gravity" if sig[12] < 0.45 else "high center of gravity"
        desc["head_height"]         = f"neck length index {sig[13]:.2f}"
        desc["hip_sway"]            = f"hip sway {'right' if sig[16] > 0.05 else 'left' if sig[16] < -0.05 else 'centered'}"

        return {
            "track_id":   track_id,
            "descriptors": desc,
            "raw_signature": sig.tolist(),
            "variance":    round(gt.variance or 0, 4),
            "frame_count": gt.frame_count,
        }


# ── Global instance ───────────────────────────────────────────────
_engine: Optional[GaitEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> GaitEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = GaitEngine()
        return _engine


def process_frame(image_path, detections, camera_id, ts=None):
    return get_engine().process_frame(image_path, detections, camera_id, ts)

def get_gait_profiles():
    return get_engine().get_profiles()

def analyze_gait(track_id: int):
    return get_engine().analyze_gait_features(track_id)

def label_gait_profile(gait_id: int, label: str) -> bool:
    eng = get_engine()
    with eng._lock:
        for prof in eng._profiles:
            if prof.id == gait_id:
                prof.label = label.upper()
                return True
    return False
