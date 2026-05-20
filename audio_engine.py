#!/usr/bin/env python3.12
"""
PALM COMMAND — Audio event classification via YAMNet + Silero-VAD.

YAMNet identifies sound events (dog bark, glass break, sirens, alarms, etc.)
without recording or transcribing speech (compliant with CA two-party consent).

Silero-VAD front-end skips silent regions before YAMNet runs, saving compute.

Module-level singleton for YAMNet model (lazy-init on first use).
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

# Lazy-loaded models (initialized once on first use)
_yamnet_model = None
_yamnet_class_names: Optional[list[str]] = None
_silero_vad_model = None
_silero_utils = None

# Allowlist of "interesting" event classes from YAMNet
# (filter out uninformative events like silence, background noise)
DEFAULT_YAMNET_ALLOWLIST = {
    "dog",
    "bark",
    "glass",
    "break",
    "smoke",
    "alarm",
    "siren",
    "whistle",
    "crying",
    "gunshot",
    "explosion",
    "crash",
    "slam",
    "door",
    "knock",
}


def _get_yamnet_model():
    """Lazy-load YAMNet model from TensorFlow Hub."""
    global _yamnet_model
    if _yamnet_model is None:
        try:
            import tensorflow as tf
            import tensorflow_hub as hub

            print("[audio] Loading YAMNet model from TensorFlow Hub...", flush=True)
            model_url = "https://tfhub.dev/google/yamnet/1"
            _yamnet_model = hub.load(model_url)
            print("[audio] YAMNet model loaded.", flush=True)
        except Exception as e:
            print(f"[audio] Failed to load YAMNet: {e}", flush=True)
            raise
    return _yamnet_model


def _get_silero_vad():
    """Lazy-load Silero-VAD model and utilities. Returns (None, None) on failure
    so the caller can fall back to whole-audio classification."""
    global _silero_vad_model, _silero_utils
    if _silero_vad_model is None or _silero_utils is None:
        try:
            import torch

            print("[audio] Loading Silero-VAD model...", flush=True)
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
                trust_repo=True,
            )
            _silero_vad_model = model
            _silero_utils = utils
            print("[audio] Silero-VAD model loaded.", flush=True)
        except Exception as e:
            print(f"[audio] Silero-VAD unavailable ({e}); falling back to whole-audio YAMNet", flush=True)
            return None, None
    return _silero_vad_model, _silero_utils


def _load_wav(wav_path: Path, sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Load WAV file and resample to 16 kHz if needed.
    Returns (waveform, sample_rate).
    """
    try:
        import librosa

        audio, orig_sr = librosa.load(str(wav_path), sr=sr, mono=True)
        return audio, sr
    except ImportError:
        # Fallback: try scipy
        try:
            import scipy.io.wavfile as wavfile

            sr_actual, audio = wavfile.read(str(wav_path))
            if sr_actual != sr:
                # Simple downsampling fallback (not high quality, but works)
                ratio = sr_actual / sr
                indices = np.arange(0, len(audio), ratio).astype(int)
                audio = audio[indices].astype(np.float32) / 32768.0
            else:
                audio = audio.astype(np.float32) / 32768.0
            return audio, sr
        except Exception as e:
            print(f"[audio] Failed to load WAV {wav_path}: {e}", flush=True)
            raise


def classify_audio(
    wav_path: Path,
    confidence_threshold: float = 0.3,
    allowlist: Optional[set[str]] = None,
) -> list[dict]:
    """
    Classify audio events in a WAV file using YAMNet.

    Args:
        wav_path: Path to WAV file.
        confidence_threshold: Confidence floor (0.0–1.0, default 0.3).
        allowlist: Set of interesting class names to keep. If None, uses DEFAULT_YAMNET_ALLOWLIST.

    Returns:
        List of events: [
            {
                "class_name": "dog",
                "confidence": 0.87,
                "start_s": 1.5,
                "end_s": 2.3,
            },
            ...
        ]
        Sorted by start time, filtered by confidence threshold + allowlist.
    """
    if allowlist is None:
        allowlist = DEFAULT_YAMNET_ALLOWLIST

    wav_path = Path(wav_path)
    if not wav_path.exists():
        print(f"[audio] WAV file not found: {wav_path}", flush=True)
        return []

    try:
        # Step 1: Load audio
        audio, sr = _load_wav(wav_path, sr=16000)
        if len(audio) == 0:
            return []

        # Step 2: VAD is for *speech*, which would skip the very sounds we
        # want (bark / glass / siren). Use VAD only as an optional silence
        # filter; if it returns nothing OR is unavailable, fall back to
        # classifying the entire audio span. Also do a cheap RMS check for
        # near-silent files to avoid wasted YAMNet inference.
        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        if rms < 1e-3:
            print(f"[audio] {wav_path.name} is essentially silent (rms={rms:.4f}); skipping", flush=True)
            return []

        vad_model, vad_utils = _get_silero_vad()
        merged_segments: list[dict] = []
        if vad_model is not None and vad_utils is not None:
            try:
                get_speech_timestamps = vad_utils[0]
                speech_timestamps = get_speech_timestamps(
                    audio, vad_model, sampling_rate=sr, return_seconds=True
                )
                for seg in speech_timestamps:
                    if merged_segments and seg["start"] - merged_segments[-1]["end"] < 0.5:
                        merged_segments[-1]["end"] = seg["end"]
                    else:
                        merged_segments.append(dict(seg))
            except Exception as e:
                print(f"[audio] VAD inference failed ({e}); using whole audio", flush=True)
                merged_segments = []

        # Fallback: classify the whole clip (most common case for non-speech
        # events like barks / glass breaks).
        if not merged_segments:
            merged_segments = [{"start": 0.0, "end": len(audio) / sr}]

        print(f"[audio] Classifying {len(merged_segments)} segment(s)", flush=True)

        # Step 3: YAMNet inference on speech segments
        yamnet = _get_yamnet_model()
        class_names = _get_yamnet_class_names()

        events = []
        for seg in merged_segments:
            start_idx = int(seg["start"] * sr)
            end_idx = int(seg["end"] * sr)
            segment_audio = audio[start_idx:end_idx]

            if len(segment_audio) < sr // 2:
                # Skip very short segments (< 0.5s)
                continue

            # YAMNet expects audio input
            try:
                import tensorflow as tf

                # Convert to tensor and run inference
                scores, embeddings, spectrogram = yamnet(
                    tf.constant(segment_audio, dtype=tf.float32)
                )
                scores_np = scores.numpy()

                # Get event frames (YAMNet outputs @ ~10Hz, so ~100ms per frame)
                frame_duration = len(segment_audio) / sr / len(scores_np)

                for frame_idx, frame_scores in enumerate(scores_np):
                    max_class_idx = np.argmax(frame_scores)
                    max_score = float(frame_scores[max_class_idx])

                    if max_score >= confidence_threshold:
                        class_name = class_names[max_class_idx]

                        # Filter by allowlist
                        if not _matches_allowlist(class_name, allowlist):
                            continue

                        frame_start = seg["start"] + (frame_idx * frame_duration)
                        frame_end = frame_start + frame_duration

                        events.append(
                            {
                                "class_name": class_name,
                                "confidence": max_score,
                                "start_s": round(frame_start, 2),
                                "end_s": round(frame_end, 2),
                            }
                        )
            except Exception as e:
                print(f"[audio] YAMNet inference error on segment: {e}", flush=True)
                continue

        # Sort by start time and deduplicate consecutive identical events
        events.sort(key=lambda e: e["start_s"])
        events = _deduplicate_events(events)

        print(f"[audio] Classified {len(events)} events above {confidence_threshold} threshold", flush=True)
        return events

    except Exception as e:
        print(f"[audio] classify_audio failed: {e}", flush=True)
        return []


def _get_yamnet_class_names() -> list[str]:
    """Return YAMNet's 521 class display names, sourced from the model's
    bundled class_map.csv. Loaded once and cached at module scope.

    YAMNet outputs 521-dim logits; indexing a shorter list would crash on
    any class outside the truncated range. The class map CSV ships with the
    TF Hub model and is exposed via `model.class_map_path()`.
    """
    global _yamnet_class_names
    if _yamnet_class_names is not None:
        return _yamnet_class_names

    import csv
    model = _get_yamnet_model()
    csv_path = model.class_map_path().numpy().decode("utf-8")
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header: index,mid,display_name
        names = [row[2] for row in reader]

    if len(names) != 521:
        # Sanity check: real YAMNet always has 521 classes. If this diverges,
        # we want to know — but don't crash the audio pipeline over it.
        print(f"[audio] WARNING: YAMNet class map has {len(names)} entries, expected 521", flush=True)

    _yamnet_class_names = names
    print(f"[audio] Loaded {len(names)} YAMNet class names.", flush=True)
    return _yamnet_class_names


def _matches_allowlist(class_name: str, allowlist: set[str]) -> bool:
    """Check if a class name (or substring) matches the allowlist."""
    class_lower = class_name.lower()
    for allowed in allowlist:
        if allowed.lower() in class_lower or class_lower in allowed.lower():
            return True
    return False


def _deduplicate_events(events: list[dict], merge_window_s: float = 0.5) -> list[dict]:
    """
    Merge consecutive events of the same class within a time window.
    Avoids spam from overlapping frames with the same label.
    """
    if not events:
        return []

    dedup = [events[0]]
    for event in events[1:]:
        last = dedup[-1]
        if (
            event["class_name"] == last["class_name"]
            and event["start_s"] - last["end_s"] < merge_window_s
        ):
            # Merge: extend the end time, keep max confidence
            last["end_s"] = event["end_s"]
            last["confidence"] = max(last["confidence"], event["confidence"])
        else:
            dedup.append(event)

    return dedup
