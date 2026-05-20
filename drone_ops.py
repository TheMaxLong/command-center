#!/usr/bin/env python3.12
"""
COMMAND CENTER — Drone operations planner.

This module intentionally keeps real aircraft control behind a bridge boundary.
The dashboard can plan, approve, simulate, and audit property missions now; a DJI
Mobile SDK / Dock bridge can later consume the active mission payload.
"""
import json
import os
import time
from pathlib import Path
from typing import Optional

STATE_FILE = Path(os.environ.get("DRONE_STATE_FILE", "/tmp/palm_command_drone.json"))

MISSION_TEMPLATES = {
    "perimeter": {
        "id": "perimeter",
        "label": "PERIMETER CHECK",
        "summary": "Orbit property boundary and inspect gates, approaches, and fence line.",
        "altitude_ft": 120,
        "speed_mph": 8,
        "duration_min": 9,
        "route": ["HOME", "NORTH GATE", "EAST FENCE", "REAR LINE", "WEST DRIVE", "HOME"],
        "capture": ["4K VIDEO", "STILLS AT WAYPOINTS", "THERMAL IF AVAILABLE"],
    },
    "recon": {
        "id": "recon",
        "label": "PROPERTY RECON",
        "summary": "Wider pass around property approaches with attention to activity changes.",
        "altitude_ft": 180,
        "speed_mph": 12,
        "duration_min": 14,
        "route": ["HOME", "DRIVE APPROACH", "STREET EDGE", "REAR APPROACH", "ROOF LINE", "HOME"],
        "capture": ["4K VIDEO", "OBLIQUE STILLS", "AI EVENT REVIEW"],
    },
    "incident": {
        "id": "incident",
        "label": "INCIDENT LOOK",
        "summary": "Short operator-approved look at the latest camera attention area.",
        "altitude_ft": 90,
        "speed_mph": 6,
        "duration_min": 5,
        "route": ["HOME", "ATTENTION AREA", "HOLD 30S", "RETURN"],
        "capture": ["LIVE VIDEO", "STILL BURST"],
    },
}


def _default_state() -> dict:
    return {
        "bridge": {
            "status": os.environ.get("DRONE_BRIDGE_STATUS", "not_configured"),
            "provider": os.environ.get("DRONE_PROVIDER", "DJI"),
            "control_mode": os.environ.get("DRONE_CONTROL_MODE", "planner_only"),
            "note": "Connect a DJI Android Mobile SDK app, DJI Dock Cloud API bridge, or RTSP payload feed to enable live control.",
        },
        "aircraft": {
            "model": os.environ.get("DRONE_MODEL", "DJI candidate"),
            "callsign": os.environ.get("DRONE_CALLSIGN", "PALM-AIR-1"),
            "battery_pct": None,
            "link": "offline",
            "home_locked": False,
            "video": None,
        },
        "active_mission": None,
        "history": [],
    }


def _load() -> dict:
    if not STATE_FILE.exists():
        return _default_state()
    try:
        state = json.loads(STATE_FILE.read_text())
    except Exception:
        state = _default_state()
    base = _default_state()
    base.update(state if isinstance(state, dict) else {})
    return base


def _save(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def status() -> dict:
    state = _load()
    mission = state.get("active_mission")
    if mission:
        elapsed = max(0, int(time.time() - float(mission.get("started_ts", time.time()))))
        total = max(60, int(float(mission.get("duration_min", 1)) * 60))
        mission["elapsed_s"] = elapsed
        mission["progress_pct"] = min(100, int((elapsed / total) * 100))
        if mission["progress_pct"] >= 100 and mission.get("state") == "running":
            mission["state"] = "complete"
            mission["completed_ts"] = time.time()
            state.setdefault("history", []).insert(0, mission)
            state["history"] = state["history"][:25]
            state["active_mission"] = None
            _save(state)
    state["templates"] = list(MISSION_TEMPLATES.values())
    return state


def start_mission(kind: str, operator: str = "dashboard", notes: str = "") -> dict:
    state = status()
    if state.get("active_mission"):
        return {"ok": False, "error": "mission already active", "active_mission": state["active_mission"]}
    template = MISSION_TEMPLATES.get(kind)
    if not template:
        return {"ok": False, "error": f"unknown mission template: {kind}"}
    mission = {
        **template,
        "state": "running",
        "mode": state.get("bridge", {}).get("control_mode", "planner_only"),
        "started_ts": time.time(),
        "operator": operator[:40],
        "notes": notes[:200],
        "requires_remote_pilot": True,
        "constraints": [
            "Maintain visual line of sight unless operating under an approved waiver.",
            "Keep route inside owned/authorized property and away from people, roads, and neighbors' private areas.",
            "Pilot can pause, abort, or take manual control at any time.",
        ],
    }
    state["active_mission"] = mission
    _save(state)
    return {"ok": True, "mission": mission, "bridge": state.get("bridge", {})}


def abort_mission(reason: Optional[str] = None) -> dict:
    state = status()
    mission = state.get("active_mission")
    if not mission:
        return {"ok": False, "error": "no active mission"}
    mission["state"] = "aborted"
    mission["aborted_ts"] = time.time()
    mission["abort_reason"] = (reason or "operator abort")[:160]
    state.setdefault("history", []).insert(0, mission)
    state["history"] = state["history"][:25]
    state["active_mission"] = None
    _save(state)
    return {"ok": True, "mission": mission}
