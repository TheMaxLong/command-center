#!/usr/bin/env python3.12
"""
FreeMoCap Worker — Background processor for mocap pipeline.
Monitors mocap-in/ for uploads, processes with FreeMoCap, outputs to mocap-out/.
"""

import os
import sys
import json
import time
import threading
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

# Seagate drive paths
SEAGATE_BASE = Path("/Volumes/Seagate Portable Drive/command-center")
MOCAP_IN = SEAGATE_BASE / "mocap-in"
MOCAP_OUT = SEAGATE_BASE / "mocap-out"
MOCAP_FROM_DOORBELL = SEAGATE_BASE / "mocap-from-doorbell"

# FreeMoCap venv
FREEMOCAP_VENV = Path.home() / "freemocap" / ".venv"
FREEMOCAP_BIN = FREEMOCAP_VENV / "bin" / "python"

# Ensure paths exist
MOCAP_IN.mkdir(parents=True, exist_ok=True)
MOCAP_OUT.mkdir(parents=True, exist_ok=True)
MOCAP_FROM_DOORBELL.mkdir(parents=True, exist_ok=True)


@dataclass
class MocapJob:
    """Metadata for a mocap processing job."""
    job_id: str
    input_file: str
    source: str  # 'drag-drop', 'doorbell', etc.
    status: str  # 'queued', 'processing', 'done', 'error'
    started_at: str = ""
    completed_at: str = ""
    error: str = ""
    output_dir: str = ""


class MocapWorker:
    """Background worker for FreeMoCap processing."""

    def __init__(self):
        self.job_queue = []
        self.job_cache = {}  # job_id -> MocapJob
        self.lock = threading.Lock()
        self.running = False
        self.worker_thread = None

    def start(self):
        """Start the background worker thread."""
        if self.running:
            return
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.worker_thread.start()
        print("[FreeMoCap Worker] Started", flush=True)

    def stop(self):
        """Stop the background worker thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        print("[FreeMoCap Worker] Stopped", flush=True)

    def submit_job(self, input_file: Path, source: str = "drag-drop") -> str:
        """Queue a file for processing. Returns job_id."""
        job_id = f"mocap_{datetime.now().isoformat()}"
        job = MocapJob(
            job_id=job_id,
            input_file=str(input_file),
            source=source,
            status="queued"
        )
        with self.lock:
            self.job_queue.append(job)
            self.job_cache[job_id] = job
        print(f"[FreeMoCap Worker] Queued job {job_id}: {input_file}", flush=True)
        return job_id

    def get_job_status(self, job_id: str) -> dict:
        """Retrieve job metadata."""
        with self.lock:
            if job_id not in self.job_cache:
                return {"error": "job not found"}
            job = self.job_cache[job_id]
            return asdict(job)

    def _process_loop(self):
        """Background loop: dequeue and process jobs."""
        while self.running:
            with self.lock:
                if not self.job_queue:
                    # Idle
                    pass
                else:
                    job = self.job_queue.pop(0)

            if not self.running:
                break

            if job and job.status == "queued":
                self._process_job(job)

            time.sleep(1)

    def _process_job(self, job: MocapJob):
        """Execute FreeMoCap on a single input file."""
        input_path = Path(job.input_file)
        if not input_path.exists():
            job.status = "error"
            job.error = f"Input file not found: {input_path}"
            print(f"[FreeMoCap Worker] ERROR {job.job_id}: {job.error}", flush=True)
            return

        output_dir = MOCAP_OUT / job.job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # FreeMoCap requires a recording folder with synchronized_videos/ subdirectory
        recording_dir = output_dir / "recording"
        recording_dir.mkdir(parents=True, exist_ok=True)
        sync_videos_dir = recording_dir / "synchronized_videos"
        sync_videos_dir.mkdir(parents=True, exist_ok=True)

        # Copy input video into the recording structure
        video_filename = input_path.name
        video_dest = sync_videos_dir / video_filename
        try:
            shutil.copy2(input_path, video_dest)
            print(f"[FreeMoCap Worker] Copied video: {video_dest}", flush=True)
        except Exception as e:
            job.status = "error"
            job.error = f"Failed to copy video: {str(e)[:300]}"
            print(f"[FreeMoCap Worker] ERROR {job.job_id}: {job.error}", flush=True)
            return

        try:
            job.status = "processing"
            job.started_at = datetime.now().isoformat()
            print(f"[FreeMoCap Worker] Processing {job.job_id}...", flush=True)

            # Launch FreeMoCap via the launcher script using the venv Python
            script_path = Path(__file__).parent / "scripts" / "run_freemocap.py"
            result = subprocess.run(
                [
                    str(FREEMOCAP_BIN),
                    str(script_path),
                    str(recording_dir)
                ],
                capture_output=True,
                timeout=300,  # 5-minute timeout
                text=True
            )

            if result.returncode != 0:
                job.status = "error"
                stderr_msg = result.stderr[:500] if result.stderr else result.stdout[:500]
                job.error = stderr_msg
                print(f"[FreeMoCap Worker] FAILED {job.job_id}: {job.error}", flush=True)
                return

            # FreeMoCap outputs to recording_dir/output_data/
            output_data_dir = recording_dir / "output_data"
            if output_data_dir.exists():
                job.status = "done"
                job.completed_at = datetime.now().isoformat()
                job.output_dir = str(output_data_dir)
                print(f"[FreeMoCap Worker] DONE {job.job_id}: {output_data_dir}", flush=True)
            else:
                job.status = "error"
                job.error = "output_data/ directory not created (FreeMoCap may have failed silently)"
                print(f"[FreeMoCap Worker] ERROR {job.job_id}: {job.error}", flush=True)

        except subprocess.TimeoutExpired:
            job.status = "error"
            job.error = "Processing timeout (>5min)"
            print(f"[FreeMoCap Worker] TIMEOUT {job.job_id}", flush=True)
        except Exception as e:
            job.status = "error"
            job.error = str(e)[:500]
            print(f"[FreeMoCap Worker] EXCEPTION {job.job_id}: {e}", flush=True)

    def ingest_doorbell_clip(self, event_id: str, clip_path: Path, metadata: dict = None):
        """Auto-ingest a clip from the doorbell event pipeline."""
        if metadata is None:
            metadata = {}

        # Copy clip to mocap-from-doorbell/ with metadata sidecar
        output_clip = MOCAP_FROM_DOORBELL / f"{event_id}.mp4"
        output_meta = MOCAP_FROM_DOORBELL / f"{event_id}.json"

        try:
            shutil.copy2(clip_path, output_clip)
            with open(output_meta, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"[FreeMoCap Worker] Doorbell ingest: {event_id}", flush=True)

            # Queue for processing
            job_id = self.submit_job(output_clip, source="doorbell")
            return job_id
        except Exception as e:
            print(f"[FreeMoCap Worker] Doorbell ingest FAILED {event_id}: {e}", flush=True)
            return None


# Singleton instance
_worker = None


def get_worker() -> MocapWorker:
    """Get or create the singleton worker."""
    global _worker
    if _worker is None:
        _worker = MocapWorker()
    return _worker


if __name__ == "__main__":
    worker = get_worker()
    worker.start()
    print("FreeMoCap Worker running (daemon)... Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        worker.stop()
