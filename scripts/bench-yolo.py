#!/usr/bin/env python3
"""
bench-yolo.py — compare YOLO model variants on archived doorbell snapshots.

For each model, runs inference on N sampled frames and reports:
  - First-load time (model download + warmup)
  - Per-image median + mean inference time
  - Detection count (all classes)
  - Person detection count specifically
  - Per-person confidence distribution

Designed to inform the PLAN Phase 1.1 question: should we swap yolov8s.pt for
yolov10n.pt or yolov10s.pt? This script does NOT modify ai_engine.py — it just
reports. Max picks.

Run inside the vision-watcher container:
  docker exec palm-vision-watcher python3 /app/scripts/bench-yolo.py

Output: /tmp/yolo-bench.json + summary printed to stdout.
"""
import glob
import json
import os
import random
import statistics
import sys
import time
from typing import Any

MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov10n.pt", "yolov10s.pt"]
SAMPLE_DIR = "/tmp/cams/doorbell/archive"
SAMPLE_SIZE = 15
SEED = 42
CONF = float(os.environ.get("AI_MIN_CONF", "0.30"))


def main() -> int:
    random.seed(SEED)
    all_frames = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.jpg")))
    if not all_frames:
        print(f"no frames at {SAMPLE_DIR}", file=sys.stderr)
        return 1
    frames = random.sample(all_frames, min(SAMPLE_SIZE, len(all_frames)))
    print(f"bench: {len(frames)} frames, {len(MODELS)} models, conf>={CONF}\n", flush=True)

    from ultralytics import YOLO

    results: dict[str, dict[str, Any]] = {}

    for model_name in MODELS:
        print(f"\n=== {model_name} ===", flush=True)
        load_start = time.time()
        try:
            model = YOLO(model_name)
        except Exception as e:
            print(f"  ! failed to load: {e}", flush=True)
            results[model_name] = {"error": str(e)}
            continue
        # Warmup pass
        _ = model.predict(frames[0], conf=CONF, verbose=False)
        load_time = time.time() - load_start

        per_frame_ms: list[float] = []
        det_counts: list[int] = []
        person_confs: list[float] = []
        person_counts: list[int] = []

        for f in frames:
            t0 = time.time()
            res = model.predict(f, conf=CONF, verbose=False)[0]
            per_frame_ms.append((time.time() - t0) * 1000)
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                det_counts.append(0)
                person_counts.append(0)
                continue
            det_counts.append(int(len(boxes)))
            classes = [model.names[int(c)] for c in boxes.cls.tolist()]
            confs = boxes.conf.tolist()
            persons = [c for cls, c in zip(classes, confs) if cls == "person"]
            person_counts.append(len(persons))
            person_confs.extend(persons)

        summary = {
            "load_warmup_s": round(load_time, 2),
            "infer_ms_median": round(statistics.median(per_frame_ms), 1),
            "infer_ms_mean": round(statistics.mean(per_frame_ms), 1),
            "infer_ms_max": round(max(per_frame_ms), 1),
            "detections_total": sum(det_counts),
            "detections_mean_per_frame": round(statistics.mean(det_counts), 2),
            "person_detections_total": sum(person_counts),
            "person_conf_mean": round(statistics.mean(person_confs), 3) if person_confs else None,
            "person_conf_median": round(statistics.median(person_confs), 3) if person_confs else None,
            "frames_with_person": sum(1 for c in person_counts if c > 0),
        }
        results[model_name] = summary
        for k, v in summary.items():
            print(f"  {k}: {v}", flush=True)

    out_path = "/tmp/yolo-bench.json"
    with open(out_path, "w") as fh:
        json.dump({"frames_sampled": frames, "conf": CONF, "results": results}, fh, indent=2)
    print(f"\nwrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
