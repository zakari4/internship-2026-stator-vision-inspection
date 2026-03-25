import cv2
import time
import numpy as np
import sys
import os
import gc

# Add the project directory instead of the server direction to the python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from server.inference import ModelManager

print("Initializing ModelManager...")
manager = ModelManager()

# Create a dummy frame
frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

def test_fps(model_name, label):
    print(f"\n--- Testing {label}: {model_name} ---")
    if not manager.load_model(model_name):
        print("Failed to load model.")
        return
        
    print("Warming up...")
    for _ in range(3):
        manager.predict(frame)
        
    print("Measuring FPS...")
    num_frames = 20
    start = time.time()
    for _ in range(num_frames):
        manager.predict(frame)
    end = time.time()
    
    fps = num_frames / (end - start)
    print(f"FPS: {fps:.2f} ({(end-start)/num_frames*1000:.1f}ms per frame)")

test_fps("yolov8m_seg", "YOLO Model (FP16)")

gc.collect()

test_fps("unet_resnet18", "PyTorch Model (FP16+Compile)")
