#!/usr/bin/env python3
"""
Quick test script for YOLOv11 object detection
Usage: python test_detection.py [path_to_image_or_video]
"""

from ultralytics import YOLO
import cv2
import sys

# Load the trained model
MODEL_PATH = 'runs/detect/yolo11n_training5/weights/best.pt'
model = YOLO(MODEL_PATH)

print(f"Loaded model: {MODEL_PATH}")
print("=" * 50)

# Test on image or video
if len(sys.argv) > 1:
    source = sys.argv[1]
else:
    # Default: use webcam
    source = 0
    print("No input specified, using webcam (press 'q' to quit)")

# Run inference
results = model.predict(
    source=source,
    conf=0.5,      # Confidence threshold
    iou=0.45,      # NMS IOU threshold
    show=True,     # Display results
    save=True,     # Save results to runs/detect/predict
    stream=True    # Stream mode for video/webcam
)

# Process results
for i, r in enumerate(results):
    print(f"\nFrame {i+1}:")
    print(f"  Detected {len(r.boxes)} objects")
    
    # Print each detection
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names[cls]
        print(f"    - {name}: {conf:.2f}")
    
    # For webcam, press 'q' to quit
    if source == 0 and cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("\n" + "=" * 50)
print("Detection completed!")
print(f"Results saved to: runs/detect/predict/")
