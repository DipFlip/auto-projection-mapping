# Projection Mapping

Depth-based projection mapping using a NexiGo Nova Mini projector's camera.

## Quick Start

```bash
source .venv/bin/activate
python server.py
```

- **Control panel:** http://localhost:8000
- **Projection view:** http://localhost:8000/projection

## Projector Setup

```bash
# Connect to projector via ADB over WiFi
adb connect 192.168.4.194:5555

# Start IP Webcam on projector
adb shell monkey -p com.pas.webcam -c android.intent.category.LAUNCHER 1

# Open projection view on projector
adb shell am start -a android.intent.action.VIEW -d "http://YOUR_MAC_IP:8000/projection"
```

Camera endpoint: http://192.168.4.194:8080/photo.jpg

## Setup from Scratch

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install torch torchvision awesome-depth-anything-3 fastapi uvicorn requests
```
