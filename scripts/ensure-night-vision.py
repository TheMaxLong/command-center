#!/usr/bin/env python3
"""
ensure-night-vision.py — idempotent guard on the Tapo D210 night vision setting.

Why this exists: 2026-05-19 the doorbell's night_vision_mode was found stuck on
`dbl_night_vision` with an inverted 6AM-6PM schedule, producing 0.7/255 black
frames at night. Fixed by switching to `wtl_night_vision`. We don't know what
caused the drift (firmware reset? user app interaction?), so this script
periodically re-asserts the expected mode.

Behavior:
  - Reads current night_vision_mode via pytapo.
  - If already correct, exits 0 silently (no log noise on no-op).
  - If drifted, applies the correct mode and logs the before/after.
  - If camera is unreachable (asleep), exits 0 (NOT fatal — battery doorbells
    sleep between motion events; we'll catch it on a later run).
  - All other failures exit 1 and log the traceback.

Run via:
  - start.sh: backgrounded after docker compose up (best-effort)
  - launchd: every hour as belt-and-suspenders
  - manual: `python3 ~/palm-command/scripts/ensure-night-vision.py`

Env:
  TAPO_IP        (required — pulled from ~/palm-command/.env if not set)
  TAPO_PASSWORD  (required — same)
  EXPECTED_MODE  (default: wtl_night_vision)
"""
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

LOG_FILE = pathlib.Path.home() / ".local" / "state" / "palm-night-vision.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

EXPECTED_MODE = os.environ.get("EXPECTED_MODE", "wtl_night_vision")


def log(level: str, **fields) -> None:
    payload = {"ts": datetime.now(timezone.utc).isoformat(), "level": level, **fields}
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(payload) + "\n")


def load_dotenv() -> None:
    env_path = pathlib.Path.home() / "palm-command" / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> int:
    load_dotenv()
    ip = os.environ.get("TAPO_IP")
    pwd = os.environ.get("TAPO_PASSWORD")
    if not ip or not pwd:
        log("error", reason="missing TAPO_IP or TAPO_PASSWORD")
        return 1

    try:
        from pytapo import Tapo
    except ImportError as e:
        log("error", reason="pytapo not installed", detail=str(e))
        return 1

    try:
        cam = Tapo(ip, "admin", pwd, pwd)
        cfg = cam.executeFunction("getNightVisionModeConfig", {"image": {"name": ["switch"]}})
    except Exception as e:
        # Camera asleep / unreachable — not fatal. Battery doorbells sleep.
        msg = str(e)
        if "Connection refused" in msg or "Connection reset" in msg or "timed out" in msg:
            return 0
        log("error", reason="getNightVisionModeConfig failed", detail=msg)
        return 1

    try:
        current = cfg.get("image", {}).get("switch", {}).get("night_vision_mode")
    except (AttributeError, KeyError):
        current = None

    if current == EXPECTED_MODE:
        # No-op. No log noise on the happy path.
        return 0

    log("warn", action="drift_detected", was=current, will_be=EXPECTED_MODE)

    try:
        cam.executeFunction(
            "setNightVisionModeConfig",
            {"image": {"switch": {"night_vision_mode": EXPECTED_MODE}}},
        )
    except Exception as e:
        log("error", reason="setNightVisionModeConfig failed", detail=str(e), was=current)
        return 1

    log("info", action="re_applied", mode=EXPECTED_MODE, was=current)
    return 0


if __name__ == "__main__":
    sys.exit(main())
