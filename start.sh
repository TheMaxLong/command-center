#!/bin/bash
cd "$(dirname "$0")"

echo ""
echo "  PALM COMMAND — startup"
echo "  ─────────────────────────────"

# Kill any leftover watcher or dashboard server
pkill -f doorbell_watcher.py 2>/dev/null
pkill -9 -f "http.server 8888" 2>/dev/null
sleep 1

# Start go2rtc in Docker
echo "  [1/3] Starting go2rtc streams..."
docker compose up -d

# Start doorbell watcher in background
echo "  [2/3] Starting doorbell watcher..."
python3.12 "$(dirname "$0")/doorbell_watcher.py" &
WATCHER_PID=$!

# Serve dashboard
echo "  [3/3] Opening dashboard..."
sleep 1
python3 -m http.server 8888 --directory dashboard &
HTTP_PID=$!

sleep 1
open http://localhost:8888

echo ""
echo "  Dashboard  : http://localhost:8888"
echo "  go2rtc UI  : http://localhost:1984"
echo "  Doorbell   : http://localhost:8181/status"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

trap "kill $HTTP_PID $WATCHER_PID 2>/dev/null; echo '  Stopped.'" EXIT
wait $HTTP_PID
