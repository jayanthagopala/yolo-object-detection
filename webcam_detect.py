#!/usr/bin/env python3
"""
Real-time YOLO Object Detection from Webcam
Optimized for Apple Silicon (M1/M2/M3/M4)

A simplified script focused on webcam detection with real-time performance monitoring.
"""

import time
import cv2
import torch
from ultralytics import YOLO


class WebcamDetector:
    def __init__(self, model_name="yolov8n.pt", camera_id=0):
        """
        Initialize webcam detector.
        
        Args:
            model_name (str): YOLO model to use
            camera_id (int): Camera device ID
        """
        self.model_name = model_name
        self.camera_id = camera_id
        self.model = None
        self.cap = None
        
        # Performance tracking
        self.frame_times = []
        self.detection_times = []
        
    def setup(self):
        """Setup model and camera."""
        print(f"🚀 Setting up YOLO detector with {self.model_name}")
        
        # Check for Apple Silicon optimization
        if torch.backends.mps.is_available():
            print("✅ MPS (Metal Performance Shaders) available - using GPU acceleration")
            device = "mps"
        else:
            print("💻 Using CPU")
            device = "cpu"
        
        # Load model
        self.model = YOLO(self.model_name)
        print(f"📦 Model loaded: {self.model_name}")
        
        # Setup camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Could not open camera {self.camera_id}")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera setup complete")
        
        # Warm up model
        self._warmup()
        
    def _warmup(self):
        """Warm up the model for consistent performance."""
        print("🔥 Warming up model...")
        dummy_frame = torch.randn(480, 640, 3).numpy() * 255
        dummy_frame = dummy_frame.astype('uint8')
        
        start_time = time.time()
        _ = self.model(dummy_frame, verbose=False)
        warmup_time = time.time() - start_time
        print(f"⚡ Warmup completed in {warmup_time:.3f}s")
        
    def run(self):
        """Run real-time detection."""
        print("\n🎥 Starting real-time detection...")
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save current frame")
        print("  - Press 'r' to reset performance stats")
        print("  - Press 'p' to print performance info")
        
        frame_count = 0
        save_count = 0
        fps_update_interval = 30
        last_fps_time = time.time()
        
        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break
                
                frame_start = time.time()
                
                # Run detection
                detection_start = time.time()
                results = self.model(frame, verbose=False)
                detection_time = time.time() - detection_start
                
                # Draw results
                annotated_frame = results[0].plot()
                
                # Add performance info to frame
                self._add_performance_overlay(annotated_frame, detection_time)
                
                # Calculate frame time
                frame_time = time.time() - frame_start
                self.frame_times.append(frame_time)
                self.detection_times.append(detection_time)
                
                # Keep only recent times (for rolling average)
                if len(self.frame_times) > 100:
                    self.frame_times.pop(0)
                    self.detection_times.pop(0)
                
                # Display frame
                cv2.imshow('YOLO Real-time Detection', annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    save_path = f"webcam_detection_{save_count:04d}.jpg"
                    cv2.imwrite(save_path, annotated_frame)
                    print(f"💾 Frame saved: {save_path}")
                    save_count += 1
                elif key == ord('r'):
                    # Reset performance stats
                    self.frame_times.clear()
                    self.detection_times.clear()
                    print("🔄 Performance stats reset")
                elif key == ord('p'):
                    # Print performance info
                    self._print_performance_stats()
                
                frame_count += 1
                
                # Print FPS every N frames
                if frame_count % fps_update_interval == 0:
                    current_time = time.time()
                    fps = fps_update_interval / (current_time - last_fps_time)
                    last_fps_time = current_time
                    print(f"📊 FPS: {fps:.1f} | Avg Detection: {detection_time*1000:.1f}ms")
                    
        except KeyboardInterrupt:
            print("\n⏹️  Detection stopped by user")
        finally:
            self.cleanup()
            
    def _add_performance_overlay(self, frame, detection_time):
        """Add performance information overlay to frame."""
        if len(self.frame_times) > 10:  # Only show after collecting some data
            avg_frame_time = sum(self.frame_times[-10:]) / 10
            avg_detection_time = sum(self.detection_times[-10:]) / 10
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            # Add text overlay
            overlay_text = [
                f"FPS: {fps:.1f}",
                f"Detection: {avg_detection_time*1000:.1f}ms",
                f"Total: {avg_frame_time*1000:.1f}ms"
            ]
            
            y_offset = 30
            for i, text in enumerate(overlay_text):
                cv2.putText(frame, text, (10, y_offset + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                           
    def _print_performance_stats(self):
        """Print detailed performance statistics."""
        if not self.frame_times:
            print("📊 No performance data available yet")
            return
            
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        avg_detection_time = sum(self.detection_times) / len(self.detection_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        print("\n📊 Performance Statistics:")
        print(f"  Average FPS: {fps:.2f}")
        print(f"  Average detection time: {avg_detection_time*1000:.2f}ms")
        print(f"  Average frame time: {avg_frame_time*1000:.2f}ms")
        print(f"  Total frames processed: {len(self.frame_times)}")
        
        # Show detection breakdown if available
        latest_result = self.model.predictor.results[-1] if hasattr(self.model, 'predictor') and self.model.predictor.results else None
        if latest_result and hasattr(latest_result, 'boxes') and latest_result.boxes is not None:
            print(f"  Objects in current frame: {len(latest_result.boxes)}")
            
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🧹 Cleanup completed")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time YOLO Webcam Detection")
    parser.add_argument("--model", default="yolov8n.pt", 
                       help="YOLO model (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    
    args = parser.parse_args()
    
    # Create and run detector
    detector = WebcamDetector(model_name=args.model, camera_id=args.camera)
    
    try:
        detector.setup()
        detector.run()
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        detector.cleanup()


if __name__ == "__main__":
    main()