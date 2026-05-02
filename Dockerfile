FROM python:3.12-slim

# ffmpeg for clip/snap extraction, libGL for OpenCV (RTSP motion)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv8n weights so the first motion event isn't slow
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY *.py .

CMD ["python3", "camera_watcher.py"]
