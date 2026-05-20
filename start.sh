#!/bin/bash
cd "$(dirname "$0")"

echo ""
echo "  COMMAND CENTER — startup"
echo "  ─────────────────────────────"

# Kill any leftover processes
pkill -f "serve_dashboard.py" 2>/dev/null
pkill -f "camera_watcher.py" 2>/dev/null
sleep 1

# Start go2rtc + vision-watcher in Docker
echo "  [1/2] Starting Docker services..."
docker compose up -d --build

# Best-effort: re-assert doorbell night vision (wtl_night_vision) in case the
# Tapo D210 reset to the buggy dbl_night_vision schedule (fixed 2026-05-19).
# Non-blocking, non-fatal — battery doorbells are usually asleep at boot.
(sleep 4 && docker exec palm-vision-watcher python3 /app/scripts/ensure-night-vision.py 2>/dev/null) &

# Start proxy/dashboard server (handles /api/ and /go2rtc/ proxying)
echo "  [2/2] Starting dashboard server..."
python3 "$(dirname "$0")/serve_dashboard.py" &
DASH_PID=$!

sleep 1
open http://localhost:8888

echo ""
echo "  Dashboard  : http://localhost:8888"
echo "  go2rtc UI  : http://localhost:1984"
echo "  Camera API : http://localhost:8181/status"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

trap "kill $DASH_PID 2>/dev/null; docker compose stop; echo '  Stopped.'" EXIT
wait $DASH_PID
