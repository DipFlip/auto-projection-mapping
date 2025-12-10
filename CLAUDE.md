# Projection Mapping

Run server: `source .venv/bin/activate && python server.py`

- Control: http://localhost:8000
- Projection: http://localhost:8000/projection
- Projector camera: http://192.168.4.194:8080/photo.jpg
- Projector IP: 192.168.4.194 (ADB: `adb connect 192.168.4.194:5555`)

Uses Depth Anything 3 (`awesome-depth-anything-3`) for depth estimation. Set `KMP_DUPLICATE_LIB_OK=TRUE` on Mac.
