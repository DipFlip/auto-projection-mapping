# Projection Mapping

Run server: `source .venv/bin/activate && python server.py`

- Control: http://localhost:8000
- Projection: http://localhost:8000/projection
- Projector camera: http://192.168.4.194:8080/photo.jpg
- Projector IP: 192.168.4.194 (ADB: `adb connect 192.168.4.194:5555`)

Uses Depth Anything 3 (`awesome-depth-anything-3`) for depth estimation. Set `KMP_DUPLICATE_LIB_OK=TRUE` on Mac.

## Calibration

Camera-projector calibration maps projector pixels to camera pixels via homography:

1. Click "Show Grid" to project 8x6 checkerboard
2. Click "Calibrate Grid" - detects 35 internal corners, computes homography
3. Remap tables (`/tmp/remap_x.npy`, `/tmp/remap_y.npy`) map every projector pixel to camera coordinates
4. `/api/capture` uses `cv2.remap()` to transform camera images to projector space
