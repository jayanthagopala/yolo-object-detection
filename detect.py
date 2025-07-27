#!/usr/bin/env python3
"""
YOLO Object Detection Script
Optimized for Apple Silicon (M1/M2/M3/M4)

Usage:
    python detect.py --source 0                    # webcam
    python detect.py --source image.jpg            # image
    python detect.py --source video.mp4            # video
    python detect.py --source path/to/images/      # directory
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO
import cv2


def check_apple_silicon():
    """Check if running on Apple Silicon and MPS is available."""
    if sys.platform == "darwin" and torch.backends.mps.is_available():
        print("✅ Apple Silicon detected with MPS support")
        return True
    else:
        print("ℹ️  Running on CPU or non-Apple Silicon")
        return False


def setup_model(model_name="yolov8n.pt", device=None):
    """
    Load YOLO model with optimal device selection.
    
    Args:
        model_name (str): Model name/path
        device (str): Device to use ('auto', 'cpu', 'mps', 'cuda')
    
    Returns:
        YOLO: Loaded model
    """
    print(f"Loading model: {model_name}")
    
    # Auto-select device if not specified
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
            print("🚀 Using MPS (Metal Performance Shaders) acceleration")
        elif torch.cuda.is_available():
            device = "cuda"
            print("🚀 Using CUDA acceleration")
        else:
            device = "cpu"
            print("💻 Using CPU")
    
    model = YOLO(model_name)
    
    # Warm up the model
    print("Warming up model...")
    dummy_img = torch.randn(1, 3, 640, 640)
    if device == "mps":
        dummy_img = dummy_img.to("mps")
    
    start_time = time.time()
    _ = model(dummy_img, verbose=False)
    warmup_time = time.time() - start_time
    print(f"Model warmup completed in {warmup_time:.3f}s")
    
    return model


def detect_image(model, source, save_dir="results"):
    """Detect objects in a single image."""
    results = model(source)
    
    # Create results directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Save results
    for i, result in enumerate(results):
        # Save annotated image
        annotated = result.plot()
        output_path = os.path.join(save_dir, f"result_{i}.jpg")
        cv2.imwrite(output_path, annotated)
        print(f"Result saved to: {output_path}")
        
        # Print detection summary
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            print(f"Detected {len(boxes)} objects:")
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                print(f"  - {class_name}: {confidence:.3f}")
        else:
            print("No objects detected")


def detect_webcam(model, camera_id=0):
    """Real-time detection from webcam."""
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    print("Starting webcam detection. Press 'q' to quit.")
    print("Press 's' to save current frame.")
    
    frame_count = 0
    fps_counter = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Run detection
        results = model(frame, verbose=False)
        
        # Draw results
        annotated_frame = results[0].plot()
        
        # Calculate FPS
        fps_counter += 1
        if fps_counter % 30 == 0:  # Update every 30 frames
            current_time = time.time()
            fps = 30 / (current_time - start_time)
            start_time = current_time
            print(f"FPS: {fps:.1f}")
        
        # Display frame
        cv2.imshow('YOLO Detection - Press q to quit, s to save', annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            os.makedirs("results", exist_ok=True)
            save_path = f"results/webcam_frame_{frame_count:04d}.jpg"
            cv2.imwrite(save_path, annotated_frame)
            print(f"Frame saved to: {save_path}")
            frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--source", default="0", help="Source: 0 for webcam, path for image/video/directory")
    parser.add_argument("--model", default="yolov8n.pt", help="Model name (yolov8n.pt, yolov8s.pt, etc.)")
    parser.add_argument("--device", default=None, help="Device: auto, cpu, mps, cuda")
    parser.add_argument("--save-dir", default="results", help="Directory to save results")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    
    args = parser.parse_args()
    
    # Check system capabilities
    check_apple_silicon()
    
    # Load model
    model = setup_model(args.model, args.device)
    
    # Set detection parameters
    model.conf = args.conf
    model.iou = args.iou
    
    # Determine source type and run detection
    if args.source == "0" or args.source.isdigit():
        # Webcam
        camera_id = int(args.source)
        detect_webcam(model, camera_id)
    elif os.path.isfile(args.source):
        # Single file
        print(f"Processing file: {args.source}")
        detect_image(model, args.source, args.save_dir)
    elif os.path.isdir(args.source):
        # Directory
        print(f"Processing directory: {args.source}")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        for file_path in Path(args.source).glob("*"):
            if file_path.suffix.lower() in image_extensions:
                print(f"Processing: {file_path}")
                detect_image(model, str(file_path), args.save_dir)
    else:
        print(f"Error: Source '{args.source}' not found or not supported")
        sys.exit(1)
    
    print("Detection completed!")


if __name__ == "__main__":
    main()