# Projection Mapping Project

## Hardware
- **Projector:** NexiGo Nova Mini (PJ08)
- **OS:** Google TV (Android 11)
- **IP Address:** 192.168.4.194

## ADB Setup
```bash
# Connect via WiFi (USB-A port is host-only, not for ADB)
adb connect 192.168.4.194:5555

# Check connection
adb devices
```

## IP Webcam
Installed IP Webcam app via ADB to access the projector's front camera over HTTP.

```bash
# Install APK
adb install com.pas.webcam_1.13.25-608_minAPI14.apk

# Launch app
adb shell monkey -p com.pas.webcam -c android.intent.category.LAUNCHER 1
```

### Camera Endpoints
- **Photo:** http://192.168.4.194:8080/photo.jpg
- **Video stream:** http://192.168.4.194:8080/video
- **Browser UI:** http://192.168.4.194:8080

## Depth Anything 3 Setup

Using `awesome-depth-anything-3` package with uv:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install torch torchvision awesome-depth-anything-3
```

**Note:** Set `KMP_DUPLICATE_LIB_OK=TRUE` to avoid OpenMP conflicts on Mac.

### Usage
```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from depth_anything_3.api import DepthAnything3
import torch

device = torch.device("mps")  # Metal on Mac
model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE")
model = model.to(device=device)

prediction = model.inference(["image.jpg"])
depth = prediction.depth[0]  # numpy array
```

Inference takes ~0.3s on Apple Silicon with Metal.

## Projection Mapping Pipeline
1. **Capture** - Grab photo from projector camera via `/photo.jpg`
2. **Depth Estimation** - Use Depth Anything 3 to extract depth/shapes
3. **Projection** - Render onto detected surfaces using three.js
