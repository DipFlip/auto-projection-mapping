import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import io
import base64
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import requests
import uvicorn

from depth_anything_3.api import DepthAnything3

app = FastAPI()

# Shared state for syncing between control panel and projection view
shared_state = {
    "depthScale": 0.5,
    "waveSpeed": 0,
    "waveAmp": 0,
    "hueShift": 0,
    "meshRes": 128,
    "gridOpacity": 0,
    "gridSpacing": 20,
    "gridWidth": 1,
    "edgeOpacity": 0,
    "edgeThreshold": 0.1,
    "photoEdgeOpacity": 0,
    "photoEdgeThreshold": 0.15,
    "hasCapture": False,
    "captureId": 0,
    "latestCapture": None
}

# Initialize model
print("Loading Depth Anything 3 model...")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE")
model = model.to(device=device)
print(f"Model loaded on {device}")

PROJECTOR_CAMERA_URL = "http://192.168.4.194:8080/photo.jpg"

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/projection")
async def projection():
    return FileResponse("static/projection.html")

@app.get("/api/state")
async def get_state():
    return JSONResponse({
        "depthScale": shared_state["depthScale"],
        "waveSpeed": shared_state["waveSpeed"],
        "waveAmp": shared_state["waveAmp"],
        "hueShift": shared_state["hueShift"],
        "meshRes": shared_state["meshRes"],
        "gridOpacity": shared_state["gridOpacity"],
        "gridSpacing": shared_state["gridSpacing"],
        "gridWidth": shared_state["gridWidth"],
        "edgeOpacity": shared_state["edgeOpacity"],
        "edgeThreshold": shared_state["edgeThreshold"],
        "photoEdgeOpacity": shared_state["photoEdgeOpacity"],
        "photoEdgeThreshold": shared_state["photoEdgeThreshold"],
        "hasCapture": shared_state["hasCapture"],
        "captureId": shared_state["captureId"]
    })

@app.post("/api/state")
async def update_state(data: dict):
    for key in ["depthScale", "waveSpeed", "waveAmp", "hueShift", "meshRes", "gridOpacity", "gridSpacing", "gridWidth", "edgeOpacity", "edgeThreshold", "photoEdgeOpacity", "photoEdgeThreshold"]:
        if key in data:
            shared_state[key] = data[key]
    return JSONResponse({"status": "ok"})

@app.get("/api/latest-capture")
async def latest_capture():
    if shared_state["latestCapture"]:
        return JSONResponse(shared_state["latestCapture"])
    raise HTTPException(status_code=404, detail="No capture available")

@app.get("/api/capture")
async def capture():
    """Capture image from projector camera and return depth map"""
    try:
        # Fetch image from projector camera
        response = requests.get(PROJECTOR_CAMERA_URL, timeout=5)
        response.raise_for_status()

        # Save temporarily
        img_path = "/tmp/capture.jpg"
        with open(img_path, "wb") as f:
            f.write(response.content)

        # Run depth estimation
        prediction = model.inference([img_path])
        depth = prediction.depth[0]

        # Normalize depth to 0-255 (invert so closer = brighter)
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_normalized = 1.0 - depth_normalized  # Invert
        depth_img = (depth_normalized * 255).astype(np.uint8)

        # Convert to base64
        depth_pil = Image.fromarray(depth_img)
        depth_buffer = io.BytesIO()
        depth_pil.save(depth_buffer, format="PNG")
        depth_b64 = base64.b64encode(depth_buffer.getvalue()).decode()

        # Also return original image as base64
        orig_b64 = base64.b64encode(response.content).decode()

        result = {
            "original": f"data:image/jpeg;base64,{orig_b64}",
            "depth": f"data:image/png;base64,{depth_b64}",
            "width": depth.shape[1],
            "height": depth.shape[0]
        }

        # Store in shared state for projection view
        shared_state["latestCapture"] = result
        shared_state["hasCapture"] = True
        shared_state["captureId"] += 1

        return JSONResponse(result)
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Failed to reach projector camera: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/depth-from-url")
async def depth_from_url(url: str):
    """Get depth map from any image URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        img_path = "/tmp/url_image.jpg"
        with open(img_path, "wb") as f:
            f.write(response.content)

        prediction = model.inference([img_path])
        depth = prediction.depth[0]

        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_normalized = 1.0 - depth_normalized  # Invert
        depth_img = (depth_normalized * 255).astype(np.uint8)

        depth_pil = Image.fromarray(depth_img)
        depth_buffer = io.BytesIO()
        depth_pil.save(depth_buffer, format="PNG")
        depth_b64 = base64.b64encode(depth_buffer.getvalue()).decode()

        orig_b64 = base64.b64encode(response.content).decode()

        return JSONResponse({
            "original": f"data:image/jpeg;base64,{orig_b64}",
            "depth": f"data:image/png;base64,{depth_b64}",
            "width": depth.shape[1],
            "height": depth.shape[0]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
