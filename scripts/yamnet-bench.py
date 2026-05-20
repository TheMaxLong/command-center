#!/usr/bin/env python3.12
"""
COMMAND CENTER — YAMNet Audio Classification Benchmark

Generates synthetic test audio (dog bark, glass break, silence) and runs
YAMNet inference to measure latency and validate detection accuracy.

Usage:
  docker exec palm-vision-watcher python3 /app/scripts/yamnet-bench.py
"""

import json
import subprocess
import sys
import time
import tempfile
from pathlib import Path

# Bench runs from /app/scripts; audio_engine is at /app/audio_engine.py.
sys.path.insert(0, "/app")

# Synthetic audio generation via ffmpeg + libav filters
# NOTE: synthetic audio rarely matches YAMNet's training data. These tests
# verify the PIPELINE runs end-to-end (load → VAD → YAMNet → output); they do
# NOT validate classification accuracy. For real accuracy, point classify_audio
# at a captured doorbell event WAV after enabling audio_events on a camera.
TEST_SOUNDS = {
    "tone_burst": {
        "filter": "sine=frequency=500:duration=2,atrim=0:2",
        "description": "500Hz sine, 2s — pipeline run, not bark-classifiable",
    },
    "noise_burst": {
        "filter": "anoisesrc=color=white:duration=1,atrim=0:1",
        "description": "1s white-noise burst — pipeline run",
    },
    "silence": {
        "filter": "anullsrc=r=16000:cl=mono,atrim=0:2",
        "description": "Pure silence (2s) — should short-circuit on RMS check",
    },
}


def generate_test_audio(sound_type: str, output_path: Path) -> bool:
    """Generate synthetic WAV file for testing."""
    if sound_type not in TEST_SOUNDS:
        print(f"Unknown sound type: {sound_type}", file=sys.stderr)
        return False

    spec = TEST_SOUNDS[sound_type]
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-f", "lavfi",
                "-i", spec["filter"],
                "-c:a", "pcm_s16le",
                "-ar", "16000",
                "-y",
                str(output_path),
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0 and output_path.exists():
            return True
        else:
            print(f"ffmpeg failed: {result.stderr.decode()}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"Audio generation failed: {e}", file=sys.stderr)
        return False


def run_benchmark() -> dict:
    """Generate test audio files and run YAMNet inference."""
    print("[yamnet-bench] Starting benchmark...")
    results = {
        "timestamp": time.time(),
        "tests": {},
        "summary": {},
    }

    try:
        import audio_engine
    except ImportError as e:
        print(f"[yamnet-bench] Failed to import audio_engine: {e}", file=sys.stderr)
        return {
            "error": f"audio_engine import failed: {e}",
            "timestamp": time.time(),
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for sound_name, spec in TEST_SOUNDS.items():
            print(f"[yamnet-bench] Testing {sound_name}...")
            wav_path = tmpdir / f"{sound_name}.wav"

            # Generate synthetic audio
            if not generate_test_audio(sound_name, wav_path):
                results["tests"][sound_name] = {
                    "error": "Failed to generate audio",
                    "status": "FAILED",
                }
                continue

            # Run YAMNet inference
            try:
                start = time.time()
                events = audio_engine.classify_audio(wav_path, confidence_threshold=0.2)
                elapsed = time.time() - start

                results["tests"][sound_name] = {
                    "description": spec["description"],
                    "wav_size_bytes": wav_path.stat().st_size,
                    "inference_time_s": round(elapsed, 3),
                    "events_detected": len(events),
                    "top_events": events[:3] if events else [],
                    "status": "OK",
                }

                # Log summary
                if events:
                    top_class = events[0]["class_name"]
                    top_conf = events[0]["confidence"]
                    print(
                        f"  ✓ {sound_name}: {len(events)} events, "
                        f"top class='{top_class}' ({top_conf:.2f}), "
                        f"inference={elapsed:.3f}s",
                        flush=True,
                    )
                else:
                    print(
                        f"  ✓ {sound_name}: no events detected, "
                        f"inference={elapsed:.3f}s",
                        flush=True,
                    )
            except Exception as e:
                print(f"  ✗ {sound_name}: {e}", file=sys.stderr, flush=True)
                results["tests"][sound_name] = {
                    "error": str(e),
                    "status": "FAILED",
                }

    # Compute summary stats
    ok_tests = [t for t in results["tests"].values() if t.get("status") == "OK"]
    if ok_tests:
        avg_time = sum(t.get("inference_time_s", 0) for t in ok_tests) / len(ok_tests)
        results["summary"] = {
            "tests_passed": len(ok_tests),
            "tests_failed": len(results["tests"]) - len(ok_tests),
            "avg_inference_time_s": round(avg_time, 3),
        }
    else:
        results["summary"] = {
            "tests_passed": 0,
            "tests_failed": len(results["tests"]),
            "error": "All tests failed",
        }

    return results


if __name__ == "__main__":
    results = run_benchmark()
    print("\n[yamnet-bench] Results:")
    print(json.dumps(results, indent=2), flush=True)

    # Exit with success if tests passed
    if results["summary"].get("tests_failed", 0) == 0 and results["summary"].get("tests_passed", 0) > 0:
        print("\n[yamnet-bench] ✓ All tests passed!", flush=True)
        sys.exit(0)
    else:
        print("\n[yamnet-bench] ✗ Some tests failed.", file=sys.stderr, flush=True)
        sys.exit(1)
