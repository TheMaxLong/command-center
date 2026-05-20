# COMMAND CENTER Vision Lab

This is the success path for adding advanced computer vision without making the live camera loop fragile.

## FreeMoCap

FreeMoCap is installed locally at:

```bash
/Users/max/freemocap
```

Run it with:

```bash
cd /Users/max/freemocap
source .venv/bin/activate
freemocap
```

Use FreeMoCap as an offline lab tool for archived clips and motion analysis. Do not put it directly in the live watcher path yet; it is better suited to post-event review than always-on surveillance.

Current note: FreeMoCap imports successfully as `v1.8.2`, but Blender was not found. Install Blender later if you want export/viewer workflows.

## PALM Live Vision Stack

The live system now has a dedicated `vision_tools.py` layer for:

- normalized bounding boxes
- attention-zone scoring
- zone occupancy
- track summaries
- operator-priority hints
- optional Supervision/Norfair capability reporting

The live Docker image intentionally does not require Supervision/Norfair yet.
They are listed in `requirements-vision-lab.txt` so experiments cannot break the camera watcher.

Endpoints:

```text
GET /api/vision/capabilities
GET /api/vision/scene/<camera_id>
GET /api/scene/zones?camera=<camera_id>
POST /api/scene/zones/<camera_id>
```

## Recommended Roadmap

1. Keep COMMAND CENTER as the operator UI and camera brain.
2. Use `supervision` for polished zone overlays, line crossing, dwell zones, and count analytics.
3. Keep the existing Kalman/ByteTrack tracker as the main live tracker; use Norfair only where it adds clear stability.
4. Add a semantic video-search sidecar later so the agent can answer requests like “show me when someone was near the stairs.”
5. Use FreeMoCap for special post-event body-motion analysis from archived clips.
