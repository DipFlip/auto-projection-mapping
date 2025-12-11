import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import io
import json
import base64
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import requests
import uvicorn
import cv2

from depth_anything_3.api import DepthAnything3

# Calibration data file
CALIBRATION_FILE = "calibration.json"

app = FastAPI()

# Load saved calibration if exists
def load_calibration():
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, 'r') as f:
            return json.load(f)
    return None

def save_calibration(data):
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(data, f)

calibration_data = load_calibration()

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
    "latestCapture": None,
    "blankScreen": False,  # Show black screen for capture
    "testMarker": None,  # Test marker position for alignment testing {"x": 0-1, "y": 0-1, "size": pixels}
    "calibrated": calibration_data is not None and "crop" in calibration_data,
    "homography": None,  # Legacy, kept for compatibility
    "showGridCalibration": None  # Grid calibration pattern config
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
    return FileResponse("static/index.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

@app.get("/projection")
async def projection():
    return FileResponse("static/projection.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

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
        "cropLeft": shared_state["cropLeft"],
        "cropRight": shared_state["cropRight"],
        "cropTop": shared_state["cropTop"],
        "cropBottom": shared_state["cropBottom"],
        "hasCapture": shared_state["hasCapture"],
        "captureId": shared_state["captureId"]
    })

@app.post("/api/state")
async def update_state(data: dict):
    for key in ["depthScale", "waveSpeed", "waveAmp", "hueShift", "meshRes", "gridOpacity", "gridSpacing", "gridWidth", "edgeOpacity", "edgeThreshold", "photoEdgeOpacity", "photoEdgeThreshold", "cropLeft", "cropRight", "cropTop", "cropBottom"]:
        if key in data:
            shared_state[key] = data[key]
    return JSONResponse({"status": "ok"})

@app.get("/api/latest-capture")
async def latest_capture():
    if shared_state["latestCapture"]:
        return JSONResponse(shared_state["latestCapture"])
    raise HTTPException(status_code=404, detail="No capture available")

@app.post("/api/push-image")
async def push_image(data: dict):
    """Push a pre-processed image directly to the projection"""
    original_b64 = data.get("original")
    depth_b64 = data.get("depth")
    width = data.get("width", 640)
    height = data.get("height", 480)

    result = {
        "original": original_b64,
        "depth": depth_b64 or original_b64,
        "width": width,
        "height": height
    }

    shared_state["latestCapture"] = result
    shared_state["hasCapture"] = True
    shared_state["captureId"] += 1

    return JSONResponse({"status": "ok", "captureId": shared_state["captureId"]})

def apply_calibration(img, calib_data):
    """Transform image from camera space to projector space"""
    if calib_data is None:
        return img

    # Use direct pixel remap if available (most accurate)
    if "remap_x_file" in calib_data and "remap_y_file" in calib_data:
        try:
            map_x = np.load(calib_data["remap_x_file"])
            map_y = np.load(calib_data["remap_y_file"])
            remapped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderValue=(0,0,0))
            return remapped
        except Exception as e:
            print(f"Remap failed: {e}, falling back to homography")

    # Use homography if available
    if "homography" in calib_data and calib_data["homography"] is not None:
        H = np.array(calib_data["homography"], dtype=np.float32)
        proj_size = calib_data.get("projector_size", [1920, 1080])
        warped = cv2.warpPerspective(img, H, (proj_size[0], proj_size[1]))
        return warped

    # Fall back to simple crop
    if "crop" not in calib_data:
        return img

    crop = calib_data["crop"]
    left = crop["left"]
    right = crop["right"]
    top = crop["top"]
    bottom = crop["bottom"]

    return img[top:bottom, left:right]

@app.get("/api/capture")
async def capture():
    """Capture image from projector camera and return depth map"""
    try:
        # Fetch image from projector camera
        response = requests.get(PROJECTOR_CAMERA_URL, timeout=5)
        response.raise_for_status()

        # Decode image
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Apply calibration (crop to projector area) if available
        if calibration_data and "crop" in calibration_data:
            img = apply_calibration(img, calibration_data)

        # Save warped image temporarily
        img_path = "/tmp/capture.jpg"
        cv2.imwrite(img_path, img)

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

        # Also return warped original image as base64
        _, orig_buffer = cv2.imencode('.jpg', img)
        orig_b64 = base64.b64encode(orig_buffer).decode()

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

PROJECTOR_IP = "192.168.4.194"

def get_local_ip():
    """Get the local LAN IP address"""
    import socket
    try:
        # Connect to projector to find which interface we'd use
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((PROJECTOR_IP, 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

@app.post("/api/open-on-projector")
async def open_on_projector(data: dict):
    """Open the projection URL or IP Webcam on the Android projector via ADB"""
    import subprocess

    action = data.get("action", "projection")  # "projection" or "camera"
    host_ip = get_local_ip()

    try:
        # First ensure ADB is connected
        connect_result = subprocess.run(
            ["adb", "connect", f"{PROJECTOR_IP}:5555"],
            capture_output=True, text=True, timeout=5
        )

        if action == "camera":
            # Launch IP Webcam app
            result = subprocess.run(
                ["adb", "-s", f"{PROJECTOR_IP}:5555", "shell", "am", "start",
                 "-n", "com.pas.webcam/.Rolling"],
                capture_output=True, text=True, timeout=10
            )
            success_msg = "Opened IP Webcam on projector"
        elif action == "refresh":
            # Send F5 key to refresh the browser
            result = subprocess.run(
                ["adb", "-s", f"{PROJECTOR_IP}:5555", "shell", "input", "keyevent", "KEYCODE_F5"],
                capture_output=True, text=True, timeout=10
            )
            success_msg = "Sent refresh to projector"
        else:
            # Open projection URL in browser
            projection_url = f"http://{host_ip}:8000/projection"
            result = subprocess.run(
                ["adb", "-s", f"{PROJECTOR_IP}:5555", "shell", "am", "start",
                 "-a", "android.intent.action.VIEW", "-d", projection_url],
                capture_output=True, text=True, timeout=10
            )
            success_msg = f"Opened {projection_url} on projector"

        if result.returncode != 0:
            return JSONResponse({
                "status": "error",
                "message": f"ADB command failed: {result.stderr}"
            }, status_code=500)

        return JSONResponse({
            "status": "ok",
            "message": success_msg
        })
    except FileNotFoundError:
        return JSONResponse({
            "status": "error",
            "message": "ADB not found. Install Android SDK platform-tools."
        }, status_code=500)
    except subprocess.TimeoutExpired:
        return JSONResponse({
            "status": "error",
            "message": "ADB command timed out. Is the projector reachable?"
        }, status_code=500)
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/api/blank-screen")
async def blank_screen(data: dict):
    """Toggle blank (black) screen on projection"""
    shared_state["blankScreen"] = data.get("blank", False)
    return JSONResponse({"status": "ok"})

@app.get("/api/calibration")
async def get_calibration():
    """Get current calibration status"""
    return JSONResponse({
        "calibrated": shared_state["calibrated"],
        "crop": calibration_data.get("crop") if calibration_data else None,
        "blankScreen": shared_state["blankScreen"],
        "testMarker": shared_state["testMarker"],
        "showGridCalibration": shared_state.get("showGridCalibration")
    })

@app.post("/api/test-marker")
async def set_test_marker(data: dict):
    """Set a test marker to project at camera coordinates (will be transformed via homography)"""
    if data.get("clear"):
        shared_state["testMarker"] = None
        return JSONResponse({"status": "ok", "message": "Marker cleared"})

    # Camera coordinates (normalized 0-1)
    cam_x = data.get("x", 0.5)
    cam_y = data.get("y", 0.5)
    size = data.get("size", 50)  # Size in pixels

    # Transform camera coords to projector coords using homography
    if shared_state["homography"]:
        H = np.array(shared_state["homography"], dtype=np.float32)
        # Homography expects normalized coords, with Y from bottom
        cam_y_flipped = 1.0 - cam_y
        cam_point = np.array([[[cam_x, cam_y_flipped]]], dtype=np.float32)
        proj_point = cv2.perspectiveTransform(cam_point, H)[0][0]
        proj_x = float(proj_point[0])
        proj_y = 1.0 - float(proj_point[1])  # Flip Y back for screen coords
    else:
        # No calibration, use camera coords directly
        proj_x = cam_x
        proj_y = cam_y

    shared_state["testMarker"] = {
        "cam_x": cam_x,
        "cam_y": cam_y,
        "proj_x": proj_x,
        "proj_y": proj_y,
        "size": size
    }

    return JSONResponse({
        "status": "ok",
        "camera": {"x": cam_x, "y": cam_y},
        "projector": {"x": proj_x, "y": proj_y},
        "size": size
    })

@app.post("/api/show-grid-calibration")
async def show_grid_calibration(data: dict):
    """Toggle showing grid calibration pattern on projection"""
    if data.get("show", False):
        shared_state["showGridCalibration"] = {
            "show": True,
            "cols": data.get("cols", 8),
            "rows": data.get("rows", 6),
            "margin": data.get("margin", 0.1)
        }
    else:
        shared_state["showGridCalibration"] = None
    return JSONResponse({"status": "ok"})

@app.post("/api/calibrate-grid")
async def calibrate_grid(data: dict):
    """Capture image with grid pattern and compute camera-to-projector mapping"""
    global calibration_data

    cols = data.get("cols", 8)
    rows = data.get("rows", 6)
    margin = data.get("margin", 0.1)

    try:
        # Fetch image from projector camera
        response = requests.get(PROJECTOR_CAMERA_URL, timeout=5)
        response.raise_for_status()

        # Decode image
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]

        # Save for debug
        cv2.imwrite("/tmp/grid_capture.jpg", img)

        # Detect checkerboard corners
        # The pattern size is (cols-1, rows-1) for internal corners
        pattern_size = (cols - 1, rows - 1)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

        if not ret:
            # Try with different flags
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, cv2.CALIB_CB_FAST_CHECK)

        if not ret:
            return JSONResponse({
                "status": "error",
                "message": f"Could not detect {pattern_size[0]}x{pattern_size[1]} checkerboard corners. Make sure the full grid is visible.",
                "debug_image": None
            }, status_code=400)

        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Compute expected projector coordinates for each corner
        # The grid is drawn with margin on each side
        # Internal corners are at positions 1 to cols-1 and 1 to rows-1
        projector_corners = []
        for row in range(rows - 1):
            for col in range(cols - 1):
                # Normalized position in projector space (0-1)
                px = margin + (col + 1) / cols * (1 - 2 * margin)
                py = margin + (row + 1) / rows * (1 - 2 * margin)
                projector_corners.append([px, py])

        projector_corners = np.array(projector_corners, dtype=np.float32)
        camera_corners = corners.reshape(-1, 2)

        # Find the bounding box of detected corners in camera space
        min_x = camera_corners[:, 0].min()
        max_x = camera_corners[:, 0].max()
        min_y = camera_corners[:, 1].min()
        max_y = camera_corners[:, 1].max()

        # The detected corners correspond to internal checkerboard intersections
        # We need to extrapolate to find the full projection area
        # The first corner is at (margin + 1/cols*(1-2*margin), margin + 1/rows*(1-2*margin))
        # The last corner is at (margin + (cols-1)/cols*(1-2*margin), margin + (rows-1)/rows*(1-2*margin))

        # Calculate the step size in camera pixels
        step_x = (max_x - min_x) / (cols - 2) if cols > 2 else (max_x - min_x)
        step_y = (max_y - min_y) / (rows - 2) if rows > 2 else (max_y - min_y)

        # Extrapolate to find the full projection corners (0,0) to (1,1) in projector space
        # First internal corner is at normalized position (margin + 1/cols*(1-2*margin))
        first_norm_x = margin + 1/cols * (1 - 2*margin)
        first_norm_y = margin + 1/rows * (1 - 2*margin)

        # Step in normalized space
        norm_step_x = (1 - 2*margin) / cols
        norm_step_y = (1 - 2*margin) / rows

        # Pixels per normalized unit
        px_per_norm_x = step_x / norm_step_x if norm_step_x > 0 else 1
        px_per_norm_y = step_y / norm_step_y if norm_step_y > 0 else 1

        # Compute crop values: where projector (0,0) and (1,1) map to in camera
        crop_left = min_x - first_norm_x * px_per_norm_x
        crop_top = min_y - first_norm_y * px_per_norm_y
        crop_right = crop_left + px_per_norm_x
        crop_bottom = crop_top + px_per_norm_y

        # Clamp to image bounds
        crop_left = max(0, int(crop_left))
        crop_top = max(0, int(crop_top))
        crop_right = min(w, int(crop_right))
        crop_bottom = min(h, int(crop_bottom))

        # Compute homography: projector -> camera
        proj_w, proj_h = 1920, 1080
        proj_pts_px = projector_corners * np.array([proj_w, proj_h])
        H, mask = cv2.findHomography(proj_pts_px, camera_corners, cv2.RANSAC, 5.0)

        # Build remap tables: for each projector pixel, find camera pixel
        proj_grid_x, proj_grid_y = np.meshgrid(np.arange(proj_w), np.arange(proj_h))
        proj_coords = np.stack([proj_grid_x.ravel(), proj_grid_y.ravel(), np.ones(proj_w * proj_h)], axis=0)
        cam_coords = H @ proj_coords
        cam_coords = cam_coords / cam_coords[2:3, :]
        map_x = cam_coords[0, :].reshape(proj_h, proj_w).astype(np.float32)
        map_y = cam_coords[1, :].reshape(proj_h, proj_w).astype(np.float32)

        # Save remap tables
        np.save('/tmp/remap_x.npy', map_x)
        np.save('/tmp/remap_y.npy', map_y)

        # Save calibration
        calibration_data = {
            "crop": {
                "left": crop_left,
                "right": crop_right,
                "top": crop_top,
                "bottom": crop_bottom
            },
            "camera_size": [w, h],
            "grid_corners": camera_corners.tolist(),
            "projector_corners": projector_corners.tolist(),
            "homography": H.tolist(),
            "projector_size": [proj_w, proj_h],
            "remap_x_file": "/tmp/remap_x.npy",
            "remap_y_file": "/tmp/remap_y.npy"
        }
        save_calibration(calibration_data)

        shared_state["calibrated"] = True
        shared_state["showGridCalibration"] = None

        # Draw detected corners for debug
        debug_img = img.copy()
        cv2.drawChessboardCorners(debug_img, pattern_size, corners, ret)
        # Draw computed crop rectangle
        cv2.rectangle(debug_img, (crop_left, crop_top), (crop_right, crop_bottom), (0, 255, 0), 2)
        _, buffer = cv2.imencode('.jpg', debug_img)
        debug_b64 = base64.b64encode(buffer).decode()

        return JSONResponse({
            "status": "ok",
            "message": f"Grid calibration successful! Detected {len(corners)} corners.",
            "crop": calibration_data["crop"],
            "debug_image": f"data:image/jpeg;base64,{debug_b64}"
        })

    except requests.RequestException as e:
        return JSONResponse({
            "status": "error",
            "message": f"Failed to reach camera: {e}"
        }, status_code=503)
    except Exception as e:
        import traceback
        return JSONResponse({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }, status_code=500)

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
