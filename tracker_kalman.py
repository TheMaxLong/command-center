#!/usr/bin/env python3.12
"""
PALM COMMAND — Kalman Filter + Hungarian Algorithm Multi-Object Tracker

A proper ByteTrack-inspired tracker replacing the naive IoU-only approach.

Algorithm:
  1. Kalman filter predicts next state for all active tracks
  2. Hungarian algorithm finds globally optimal assignment (not greedy)
  3. Appearance embeddings break ties when IoU is low
  4. Track lifecycle: TENTATIVE → CONFIRMED → LOST → REMOVED
  5. Direction computed from Kalman velocity vector (vx, vy)
  6. Re-ID: uses last N embeddings to handle brief occlusions

State vector: [cx, cy, vx, vy, w, h]
  cx, cy = center position (pixels)
  vx, vy = velocity (pixels per update interval)
  w,  h  = bounding box dimensions

Based on:
  ByteTrack (ECCV 2022) — high + low confidence track handling
  SORT (2016)          — foundational Kalman + Hungarian approach
  StrongSORT (2022)    — appearance-based re-identification
"""
from __future__ import annotations

import math
import threading
import time
from typing import Optional

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ── Kalman Filter for single track ───────────────────────────────

class KalmanTrack:
    """
    6-dimensional state Kalman filter for a single bounding box track.

    State:    [cx, cy, vx, vy, w, h]
    Measured: [cx, cy, w,  h]

    Constant-velocity motion model with measurement noise.
    """

    # Process noise scaling (how much we trust motion model)
    _Q_SCALE_POS = 1.0    # positional process noise
    _Q_SCALE_VEL = 0.01   # velocity process noise (assume near-constant)
    _Q_SCALE_SZE = 0.5    # size process noise

    # Measurement noise (pixel uncertainty in detection)
    _R_SCALE_POS = 4.0    # position measurement noise
    _R_SCALE_SZE = 8.0    # size measurement noise

    def __init__(self, cx: float, cy: float, w: float, h: float):
        # State [cx, cy, vx, vy, w, h]
        self.x = np.array([cx, cy, 0.0, 0.0, w, h], dtype=np.float64)

        # State covariance P
        self.P = np.diag([100.0, 100.0, 25.0, 25.0, 50.0, 50.0])

        # State transition F (constant velocity)
        self.F = np.eye(6)
        self.F[0, 2] = 1.0  # cx += vx
        self.F[1, 3] = 1.0  # cy += vy

        # Observation matrix H (observe cx, cy, w, h)
        self.H = np.zeros((4, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 4] = 1.0
        self.H[3, 5] = 1.0

        # Process noise Q
        self.Q = np.diag([
            self._Q_SCALE_POS, self._Q_SCALE_POS,
            self._Q_SCALE_VEL, self._Q_SCALE_VEL,
            self._Q_SCALE_SZE, self._Q_SCALE_SZE,
        ])

        # Measurement noise R
        self.R = np.diag([
            self._R_SCALE_POS, self._R_SCALE_POS,
            self._R_SCALE_SZE, self._R_SCALE_SZE,
        ])

    def predict(self) -> np.ndarray:
        """Advance state one step using the motion model."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, cx: float, cy: float, w: float, h: float) -> np.ndarray:
        """Incorporate a new measurement."""
        z = np.array([cx, cy, w, h], dtype=np.float64)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)      # Kalman gain
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x

    @property
    def bbox_pred(self) -> list[int]:
        """Predicted bounding box [x1, y1, x2, y2]."""
        cx, cy, _, _, w, h = self.x
        half_w = abs(w) / 2; half_h = abs(h) / 2
        return [int(cx - half_w), int(cy - half_h),
                int(cx + half_w), int(cy + half_h)]

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    @property
    def center(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def size(self) -> tuple[float, float]:
        return abs(float(self.x[4])), abs(float(self.x[5]))


# ── Track lifecycle ───────────────────────────────────────────────

class Track:
    """
    A single tracked person with Kalman state + appearance history.
    """
    TENTATIVE = "TENTATIVE"    # just created, not yet confirmed
    CONFIRMED = "CONFIRMED"    # seen multiple frames
    LOST      = "LOST"         # missed but might return
    REMOVED   = "REMOVED"      # expired — will be purged

    # Thresholds
    MIN_HITS_CONFIRM = 2       # frames before CONFIRMED
    MAX_MISS_FRAMES  = 10      # frames without match before LOST→REMOVED
    MAX_EMBED_HIST   = 8       # last N embeddings to average for re-ID

    def __init__(self, track_id: int, cx: float, cy: float, w: float, h: float,
                 confidence: float, ts: float, embedding: Optional[list] = None):
        self.id         = track_id
        self.kf         = KalmanTrack(cx, cy, w, h)
        self.state      = self.TENTATIVE
        self.hits       = 1
        self.miss       = 0
        self.first_ts   = ts
        self.last_ts    = ts
        self.confidence = confidence
        self.embeddings: list[list[float]] = [embedding] if embedding else []
        self.direction  = "UNKNOWN"
        self._prev_vel  = (0.0, 0.0)

    def predict(self) -> np.ndarray:
        return self.kf.predict()

    def update_match(self, cx: float, cy: float, w: float, h: float,
                     confidence: float, ts: float, embedding: Optional[list] = None):
        self.kf.update(cx, cy, w, h)
        self.hits      += 1
        self.miss       = 0
        self.last_ts    = ts
        self.confidence = confidence
        if self.hits >= self.MIN_HITS_CONFIRM:
            self.state = self.CONFIRMED
        if embedding:
            self.embeddings.append(embedding)
            if len(self.embeddings) > self.MAX_EMBED_HIST:
                self.embeddings.pop(0)
        self._update_direction()

    def mark_missed(self):
        self.miss += 1
        if self.miss > self.MAX_MISS_FRAMES:
            self.state = self.REMOVED
        elif self.state == self.CONFIRMED:
            self.state = self.LOST

    def _update_direction(self):
        """Compute direction from Kalman velocity vector."""
        vx, vy = self.kf.velocity
        speed  = math.sqrt(vx**2 + vy**2)
        if speed < 0.8:  # below 0.8 px/frame → stationary
            self.direction = "STATIONARY"
        elif vy < -0.5:  # moving up in frame (toward camera if top-mounted)
            self.direction = "APPROACHING"
        elif vy > 0.5:   # moving down in frame (toward exit / away)
            self.direction = "DEPARTING"
        elif abs(vx) > abs(vy) * 1.5:
            self.direction = "TRAVERSING"
        else:
            self.direction = "STATIONARY"

    @property
    def dwell_s(self) -> float:
        return self.last_ts - self.first_ts

    @property
    def mean_embedding(self) -> Optional[list[float]]:
        if not self.embeddings:
            return None
        arr = np.array(self.embeddings)
        mean = arr.mean(axis=0)
        norm = np.linalg.norm(mean)
        return (mean / norm).tolist() if norm > 0 else mean.tolist()


# ── Cost matrix helpers ───────────────────────────────────────────

def _iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """0 = identical, 1 = orthogonal, 2 = opposite."""
    an = np.array(a); bn = np.array(b)
    na = np.linalg.norm(an); nb = np.linalg.norm(bn)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - np.dot(an, bn) / (na * nb))


def _build_cost_matrix(tracks: list[Track], dets: list[dict],
                       appearance_weight: float = 0.3) -> np.ndarray:
    """
    Cost = (1 - IoU) * (1 - appearance_weight) + cosine_dist * appearance_weight

    IoU is computed between Kalman-predicted bbox and detected bbox.
    Appearance cost is cosine distance between embeddings (if available).
    """
    n_t = len(tracks)
    n_d = len(dets)
    cost = np.ones((n_t, n_d), dtype=np.float64)

    for ti, track in enumerate(tracks):
        pred_bb = track.kf.bbox_pred
        t_emb   = track.mean_embedding

        for di, det in enumerate(dets):
            iou = _iou(pred_bb, det["bbox"])
            iou_cost = 1.0 - iou

            if t_emb and det.get("embedding"):
                app_cost = _cosine_distance(t_emb, det["embedding"])
                cost[ti, di] = (iou_cost * (1.0 - appearance_weight)
                                + app_cost * appearance_weight)
            else:
                cost[ti, di] = iou_cost

    return cost


def _hungarian_assign(cost: np.ndarray, thresh: float = 0.7) -> tuple[dict, set, set]:
    """
    Run Hungarian algorithm on cost matrix.
    Returns:
      matches:       {track_idx: det_idx}
      unmatched_t:   set of track indices without a match
      unmatched_d:   set of det indices without a match
    """
    n_t, n_d = cost.shape
    if n_t == 0 or n_d == 0:
        return {}, set(range(n_t)), set(range(n_d))

    if _HAS_SCIPY:
        row_ind, col_ind = linear_sum_assignment(cost)
    else:
        # Greedy fallback if scipy missing
        row_ind, col_ind = _greedy_assign(cost)

    matches: dict[int, int] = {}
    unmatched_t = set(range(n_t))
    unmatched_d = set(range(n_d))

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < thresh:
            matches[r] = c
            unmatched_t.discard(r)
            unmatched_d.discard(c)

    return matches, unmatched_t, unmatched_d


def _greedy_assign(cost: np.ndarray) -> tuple[list, list]:
    """Greedy assignment fallback when scipy unavailable."""
    used_rows, used_cols = set(), set()
    rows, cols = [], []
    flat = [(cost[r, c], r, c)
            for r in range(cost.shape[0])
            for c in range(cost.shape[1])]
    flat.sort()
    for val, r, c in flat:
        if r not in used_rows and c not in used_cols:
            rows.append(r); cols.append(c)
            used_rows.add(r); used_cols.add(c)
    return rows, cols


# ── Main Tracker ──────────────────────────────────────────────────

class ByteTracker:
    """
    Multi-object tracker with Kalman filtering and Hungarian assignment.

    Implements two-stage matching inspired by ByteTrack:
      Stage 1: High-confidence detections matched against confirmed tracks
      Stage 2: Low-confidence detections matched against remaining tracks
               (catches occlusion recovery and far detections)
    """

    HIGH_CONF = 0.50   # detections above this go through stage 1
    LOW_CONF  = 0.30   # detections above this go through stage 2
    COST_HIGH = 0.65   # max cost for stage 1 match (1-IoU)
    COST_LOW  = 0.85   # max cost for stage 2 match

    def __init__(self, cam_id: str = "default"):
        self.cam_id   = cam_id
        self._tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: list[dict], ts: float) -> list[dict]:
        """
        Main entry point. Call each frame with person detections.

        Args:
            detections: list of detection dicts from ai_engine.detect()
                        Must have: bbox [x1,y1,x2,y2], confidence, class
                        Optional: embedding (list[float])
            ts:         current Unix timestamp

        Returns:
            detections enriched with: track_id, direction, dwell_s
        """
        # Predict all tracks forward one step
        for track in self._tracks:
            track.predict()

        # Separate person detections from others
        person_idx = [i for i, d in enumerate(detections) if d["class"] == "person"]
        person_dets = [detections[i] for i in person_idx]

        if not person_dets:
            # No detections: mark all active tracks as missed
            for t in self._tracks:
                if t.state != Track.REMOVED:
                    t.mark_missed()
            self._prune()
            return list(detections)

        # ── Stage 1: High-confidence detections vs confirmed tracks
        confirmed = [t for t in self._tracks if t.state == Track.CONFIRMED]
        high_dets = [(i, d) for i, d in enumerate(person_dets) if d["confidence"] >= self.HIGH_CONF]

        matches1, unmatched_t1, unmatched_d1 = {}, set(range(len(confirmed))), set(range(len(high_dets)))
        if confirmed and high_dets:
            cost1 = _build_cost_matrix(confirmed, [d for _, d in high_dets])
            matches1, unmatched_t1, unmatched_d1 = _hungarian_assign(cost1, self.COST_HIGH)

        # ── Stage 2: Remaining dets vs unmatched+lost tracks
        remaining_tracks = (
            [confirmed[i] for i in unmatched_t1] +
            [t for t in self._tracks if t.state in (Track.TENTATIVE, Track.LOST)]
        )
        low_dets = [(i, d) for i, d in enumerate(person_dets)
                    if i in unmatched_d1 or d["confidence"] < self.HIGH_CONF]

        matches2, unmatched_t2, unmatched_d2 = {}, set(range(len(remaining_tracks))), set(range(len(low_dets)))
        if remaining_tracks and low_dets:
            cost2 = _build_cost_matrix(remaining_tracks, [d for _, d in low_dets])
            matches2, unmatched_t2, unmatched_d2 = _hungarian_assign(cost2, self.COST_LOW)

        # ── Apply matches
        matched_person_indices: set[int] = set()

        for t_idx, d_idx in matches1.items():
            real_d_idx, det = high_dets[d_idx]
            self._apply_match(confirmed[t_idx], det, ts)
            matched_person_indices.add(real_d_idx)

        for t_idx, d_idx in matches2.items():
            real_d_idx, det = low_dets[d_idx]
            self._apply_match(remaining_tracks[t_idx], det, ts)
            matched_person_indices.add(real_d_idx)

        # ── Mark unmatched confirmed tracks as missed
        for i in unmatched_t1:
            if confirmed[i] not in remaining_tracks:
                confirmed[i].mark_missed()

        for i in unmatched_t2:
            remaining_tracks[i].mark_missed()

        # ── Create new tracks for completely unmatched high-conf detections
        for i, det in enumerate(person_dets):
            if i not in matched_person_indices and det["confidence"] >= self.HIGH_CONF:
                self._new_track(det, ts)

        # ── Prune dead tracks
        self._prune()

        # ── Build track lookup
        track_map = {t.id: t for t in self._tracks}

        # ── Enrich detections with tracking info
        enriched = list(detections)
        for list_idx, person_i in enumerate(person_idx):
            det = detections[person_i]
            # Find which track was matched
            track = self._find_track_for_det(det, ts)
            if track:
                enriched[person_i] = {
                    **det,
                    "track_id":  track.id,
                    "direction": track.direction,
                    "dwell_s":   round(track.dwell_s, 1),
                    "track_state": track.state,
                }
            else:
                enriched[person_i] = {
                    **det,
                    "track_id":  None,
                    "direction": "UNKNOWN",
                    "dwell_s":   0.0,
                    "track_state": "NEW",
                }

        return enriched

    def _apply_match(self, track: Track, det: dict, ts: float):
        cx = (det["bbox"][0] + det["bbox"][2]) / 2
        cy = (det["bbox"][1] + det["bbox"][3]) / 2
        w  = det["bbox"][2] - det["bbox"][0]
        h  = det["bbox"][3] - det["bbox"][1]
        track.update_match(cx, cy, w, h, det["confidence"], ts, det.get("embedding"))

    def _new_track(self, det: dict, ts: float):
        cx = (det["bbox"][0] + det["bbox"][2]) / 2
        cy = (det["bbox"][1] + det["bbox"][3]) / 2
        w  = det["bbox"][2] - det["bbox"][0]
        h  = det["bbox"][3] - det["bbox"][1]
        track = Track(
            track_id   = self._next_id,
            cx=cx, cy=cy, w=w, h=h,
            confidence = det["confidence"],
            ts         = ts,
            embedding  = det.get("embedding"),
        )
        self._next_id += 1
        self._tracks.append(track)

    def _find_track_for_det(self, det: dict, ts: float) -> Optional[Track]:
        """Find the best matching active track for a detection by bbox overlap."""
        best_iou = 0.2
        best_track = None
        for track in self._tracks:
            if track.state == Track.REMOVED:
                continue
            if abs(track.last_ts - ts) > 1.0:
                continue
            iou = _iou(track.kf.bbox_pred, det["bbox"])
            if iou > best_iou:
                best_iou  = iou
                best_track = track
        return best_track

    def _prune(self):
        self._tracks = [t for t in self._tracks if t.state != Track.REMOVED]

    def active_tracks(self) -> list[dict]:
        """Returns summary of all confirmed/tentative tracks."""
        return [
            {
                "id":        t.id,
                "state":     t.state,
                "direction": t.direction,
                "dwell_s":   round(t.dwell_s, 1),
                "hits":      t.hits,
                "confidence":round(t.confidence, 2),
                "center":    [round(t.kf.center[0], 1), round(t.kf.center[1], 1)],
                "velocity":  [round(t.kf.velocity[0], 2), round(t.kf.velocity[1], 2)],
            }
            for t in self._tracks
            if t.state in (Track.CONFIRMED, Track.TENTATIVE)
        ]


# ── Global registry (one tracker per camera) ─────────────────────

_bt_trackers: dict[str, ByteTracker] = {}
_bt_lock = threading.Lock()


def get_tracker(cam_id: str) -> ByteTracker:
    with _bt_lock:
        if cam_id not in _bt_trackers:
            _bt_trackers[cam_id] = ByteTracker(cam_id)
            print(f"[tracker] ByteTracker created for {cam_id} "
                  f"({'scipy Hungarian' if _HAS_SCIPY else 'greedy fallback'})", flush=True)
        return _bt_trackers[cam_id]
