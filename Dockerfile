FROM python:3.12-slim

# ffmpeg for clip/snap extraction
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLO weights so the first motion/pose event isn't slow
RUN python -c "from ultralytics import YOLO; YOLO('yolov8s.pt'); YOLO('yolov8n.pt'); YOLO('yolov8n-pose.pt')"

COPY *.py .

CMD ["python3", "camera_watcher.py"]
