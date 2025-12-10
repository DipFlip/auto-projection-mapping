import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from depth_anything_3.api import DepthAnything3
from PIL import Image
import numpy as np

# Use MPS (Metal) on Mac if available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal)")
else:
    device = torch.device("cpu")
    print("Using CPU")

print("Loading model...")
model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE")
model = model.to(device=device)

print("Running inference on test_capture.jpg...")
prediction = model.inference(["test_capture.jpg"])

print(f"Depth shape: {prediction.depth.shape}")

# Save depth map as image
depth = prediction.depth[0]
depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
depth_img = (depth_normalized * 255).astype(np.uint8)
Image.fromarray(depth_img).save("depth_output.png")
print("Saved depth map to depth_output.png")
