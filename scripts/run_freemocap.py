#!/usr/bin/env python3.12
"""
FreeMoCap headless launcher.
Invoked by freemocap_worker.py to process a recording folder.
"""

import sys
from pathlib import Path

# Use the FreeMoCap venv's Python to import correctly
from freemocap.core_processes.process_motion_capture_videos.process_recording_headless import process_recording_headless


def main(recording_path: str):
    """Process a recording folder with FreeMoCap headless."""
    try:
        path = Path(recording_path)
        if not path.exists():
            print(f"ERROR: Recording path not found: {recording_path}", file=sys.stderr)
            sys.exit(1)

        print(f"[FreeMoCap] Processing: {recording_path}")
        process_recording_headless(
            recording_path=recording_path,
            run_blender=False,           # Skip Blender for v1
            make_jupyter_notebook=False, # Skip notebook generation
            use_tqdm=False,              # Not in TTY
        )
        print(f"[FreeMoCap] Complete: {recording_path}")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_freemocap.py <recording_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
