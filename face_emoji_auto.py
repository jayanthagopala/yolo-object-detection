#!/usr/bin/env python3
"""
Face Detection with Automatic Emoji Selection Based on Mouth Movement
Real-time face detection that automatically picks emoji based on facial expressions

This version uses MediaPipe to detect facial landmarks and analyze mouth movements
to automatically select the most appropriate emoji from your assets.
"""

import time
import cv2
import numpy as np
import os
import argparse
import math
from pathlib import Path

# Import MediaPipe for facial landmark detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  MediaPipe not available. Install with: pip install mediapipe")


class AutoFaceEmojiDetector:
    def __init__(self, camera_id=0, assets_dir="Animoji", scale_factor=2.1):
        """
        Initialize face detector with automatic emoji selection.
        
        Args:
            camera_id (int): Camera device ID
            assets_dir (str): Directory containing emoji PNG files
            scale_factor (float): How much larger emoji should be than face
        """
        self.camera_id = camera_id
        self.assets_dir = Path(assets_dir)
        self.scale_factor = scale_factor
        self.cap = None
        
        # MediaPipe setup
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            print("✅ MediaPipe facial landmarks enabled")
        else:
            # Fallback to Haar cascades
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print("⚠️  Using basic face detection (install mediapipe for expression detection)")
        
        # Emoji management
        self.emojis = []
        self.emoji_names = []
        self.emoji_files = []
        self.current_emoji_idx = 0
        self.auto_mode = True
        
        # Expression detection parameters
        self.mouth_expressions = {
            'neutral': 0,
            'smile': 1, 
            'open': 2,
            'surprise': 3
        }
        
        # Performance tracking
        self.frame_times = []
        self.detection_times = []
        
    def setup(self):
        """Setup camera and load emoji assets."""
        print("🚀 Setting up Auto Face Emoji Detector")
        
        # Setup camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Could not open camera {self.camera_id}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        print("✅ Camera setup complete")
        
        # Load emoji assets
        self._load_emoji_assets()
        
    def _load_emoji_assets(self):
        """Load emoji assets from directory."""
        print(f"🎭 Loading emoji assets from {self.assets_dir}...")
        
        if not self.assets_dir.exists():
            raise RuntimeError(f"❌ Assets directory {self.assets_dir} not found!")
        
        # Look for PNG files in assets directory
        emoji_files = sorted(list(self.assets_dir.glob("*.png")))
        
        if not emoji_files:
            raise RuntimeError(f"❌ No PNG files found in {self.assets_dir} directory!")
        
        # Load emoji images
        for emoji_file in emoji_files:
            try:
                # Load image with alpha channel (BGRA)
                emoji_img = cv2.imread(str(emoji_file), cv2.IMREAD_UNCHANGED)
                
                if emoji_img is not None:
                    # Convert to BGRA if needed
                    if emoji_img.shape[2] == 3:
                        # Add alpha channel if missing
                        alpha = np.ones((emoji_img.shape[0], emoji_img.shape[1], 1), dtype=emoji_img.dtype) * 255
                        emoji_img = np.concatenate([emoji_img, alpha], axis=2)
                    
                    self.emojis.append(emoji_img)
                    self.emoji_files.append(emoji_file)
                    
                    # Extract name from filename (remove extension)
                    name = emoji_file.stem
                    self.emoji_names.append(name)
                    
                    print(f"✅ Loaded: {emoji_file.name}")
                else:
                    print(f"❌ Failed to load: {emoji_file.name}")
                    
            except Exception as e:
                print(f"❌ Error loading {emoji_file.name}: {e}")
        
        if not self.emojis:
            raise RuntimeError("❌ No emoji assets could be loaded!")
            
        print(f"🎉 Successfully loaded {len(self.emojis)} emoji assets")
        print("📝 Expression mapping:")
        expression_names = ['Neutral', 'Smile', 'Open Mouth', 'Surprise']
        for i, name in enumerate(self.emoji_names):
            expr = expression_names[i] if i < len(expression_names) else f"Custom {i+1}"
            print(f"  {expr}: {name}")
    
    def detect_faces_and_expressions(self, frame):
        """Detect faces and analyze facial expressions."""
        if not MEDIAPIPE_AVAILABLE:
            # Fallback to basic face detection
            return self._detect_faces_basic(frame)
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        faces = []
        expressions = []
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get face bounding box
                h, w, _ = frame.shape
                x_coords = [landmark.x * w for landmark in face_landmarks.landmark]
                y_coords = [landmark.y * h for landmark in face_landmarks.landmark]
                
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                face_width = x_max - x_min
                face_height = y_max - y_min
                
                faces.append((x_min, y_min, face_width, face_height))
                
                # Analyze mouth expression
                expression = self._analyze_mouth_expression(face_landmarks, w, h)
                expressions.append(expression)
        
        return faces, expressions
    
    def _detect_faces_basic(self, frame):
        """Basic face detection without expression analysis."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        # Return neutral expression for all faces
        expressions = ['neutral'] * len(faces)
        return faces, expressions
    
    def _analyze_mouth_expression(self, face_landmarks, width, height):
        """Analyze mouth landmarks to determine expression."""
        # Key mouth landmark indices for MediaPipe Face Mesh
        # Upper lip: 13, 14, 15, 16, 17, 18
        # Lower lip: 78, 95, 88, 178, 87, 14, 317, 402, 318, 324
        # Mouth corners: 61 (left), 291 (right)
        # Mouth center top: 13, bottom: 14
        
        landmarks = face_landmarks.landmark
        
        # Get mouth key points
        mouth_top = landmarks[13]  # Upper lip center
        mouth_bottom = landmarks[14]  # Lower lip center  
        mouth_left = landmarks[61]   # Left corner
        mouth_right = landmarks[291] # Right corner
        
        # Convert to pixel coordinates
        mouth_top_y = mouth_top.y * height
        mouth_bottom_y = mouth_bottom.y * height
        mouth_left_x = mouth_left.x * width
        mouth_right_x = mouth_right.x * width
        
        # Calculate mouth metrics
        mouth_height = abs(mouth_bottom_y - mouth_top_y)
        mouth_width = abs(mouth_right_x - mouth_left_x)
        
        # Calculate mouth aspect ratio and opening
        mouth_aspect_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
        
        # Get mouth corner positions for smile detection
        mouth_left_y = mouth_left.y * height
        mouth_right_y = mouth_right.y * height
        mouth_center_y = (mouth_top_y + mouth_bottom_y) / 2
        
        # Smile detection: corners higher than center
        smile_left = mouth_left_y < mouth_center_y - 2
        smile_right = mouth_right_y < mouth_center_y - 2
        is_smiling = smile_left and smile_right
        
        # Expression classification
        if mouth_aspect_ratio > 0.08:  # Very open mouth
            return 'surprise'
        elif mouth_aspect_ratio > 0.04:  # Moderately open mouth
            return 'open'
        elif is_smiling:  # Smile detected
            return 'smile'
        else:
            return 'neutral'
    
    def select_emoji_for_expression(self, expression):
        """Select appropriate emoji based on detected expression."""
        expression_to_index = {
            'neutral': 0,
            'smile': 1,
            'open': 2, 
            'surprise': 3
        }
        
        emoji_index = expression_to_index.get(expression, 0)
        # Ensure index is within bounds
        return min(emoji_index, len(self.emojis) - 1)
    
    def overlay_emoji(self, frame, faces, expressions):
        """Overlay appropriate emoji on detected faces."""
        if len(self.emojis) == 0 or len(faces) == 0:
            return frame
        
        for i, ((x, y, w, h), expression) in enumerate(zip(faces, expressions)):
            # Select emoji based on expression (if auto mode) or manual selection
            if self.auto_mode:
                emoji_idx = self.select_emoji_for_expression(expression)
            else:
                emoji_idx = self.current_emoji_idx
                
            emoji = self.emojis[emoji_idx]
            
            # Calculate emoji position and size
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            
            # Scale emoji to face size
            face_size = max(w, h)
            scaled_size = int(face_size * self.scale_factor)
            
            # Resize emoji with high-quality interpolation
            emoji_resized = cv2.resize(emoji, (scaled_size, scaled_size), interpolation=cv2.INTER_CUBIC)
            
            # Calculate overlay position (center on face)
            overlay_x = face_center_x - scaled_size // 2
            overlay_y = face_center_y - scaled_size // 2
            
            # Ensure overlay is within frame bounds
            overlay_x = max(0, min(overlay_x, frame.shape[1] - scaled_size))
            overlay_y = max(0, min(overlay_y, frame.shape[0] - scaled_size))
            
            # Apply emoji overlay
            self._apply_smooth_overlay(frame, emoji_resized, overlay_x, overlay_y)
        
        return frame
    
    def _apply_smooth_overlay(self, frame, overlay, x, y):
        """Apply emoji overlay with smooth alpha blending."""
        h, w = overlay.shape[:2]
        
        # Ensure overlay fits within frame
        if x + w > frame.shape[1]:
            w = frame.shape[1] - x
            overlay = overlay[:, :w]
        if y + h > frame.shape[0]:
            h = frame.shape[0] - y
            overlay = overlay[:h, :]
        
        if w <= 0 or h <= 0:
            return
        
        # Extract alpha channel and normalize
        overlay_bgr = overlay[:, :, :3]
        alpha = overlay[:, :, 3] / 255.0
        
        # Smooth alpha blending
        alpha_smooth = np.power(alpha, 0.8)
        
        # Get region of interest
        roi = frame[y:y+h, x:x+w]
        
        # Apply blending
        for c in range(3):
            roi[:, :, c] = alpha_smooth * overlay_bgr[:, :, c] + (1 - alpha_smooth) * roi[:, :, c]
    
    def run(self):
        """Run real-time face detection with automatic emoji selection."""
        print("\n🎭 Starting Auto Face Emoji Detection...")
        print("🤖 Auto mode: Emoji changes based on your expression!")
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 'a' to toggle auto/manual mode")
        print(f"  - Press '1-{len(self.emojis)}' to manually select emoji (manual mode)")
        print("  - Press 's' to save current frame")
        print("  - Press 'h' to show emoji list")
        print("  - Press 'p' to print performance info")
        
        frame_count = 0
        save_count = 0
        fps_update_interval = 30
        last_fps_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break
                
                frame_start = time.time()
                
                # Detect faces and expressions
                detection_start = time.time()
                faces, expressions = self.detect_faces_and_expressions(frame)
                detection_time = time.time() - detection_start
                
                # Overlay emojis based on expressions
                frame = self.overlay_emoji(frame, faces, expressions)
                
                # Add info overlay
                self._add_info_overlay(frame, faces, expressions, detection_time)
                
                # Calculate frame time
                frame_time = time.time() - frame_start
                self.frame_times.append(frame_time)
                self.detection_times.append(detection_time)
                
                # Keep only recent times
                if len(self.frame_times) > 100:
                    self.frame_times.pop(0)
                    self.detection_times.pop(0)
                
                # Display frame
                cv2.imshow('Auto Face Emoji Detection - Press q to quit', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('a'):
                    # Toggle auto/manual mode
                    self.auto_mode = not self.auto_mode
                    mode = "AUTO" if self.auto_mode else "MANUAL"
                    print(f"🔄 Switched to {mode} mode")
                elif key >= ord('1') and key <= ord('9'):
                    # Manual emoji selection
                    emoji_num = key - ord('1')
                    if emoji_num < len(self.emojis):
                        self.current_emoji_idx = emoji_num
                        if not self.auto_mode:
                            print(f"🎭 Manual emoji: {self.emoji_names[emoji_num]}")
                elif key == ord('s'):
                    # Save current frame
                    save_path = f"auto_face_emoji_{save_count:04d}.jpg"
                    cv2.imwrite(save_path, frame)
                    print(f"💾 Frame saved: {save_path}")
                    save_count += 1
                elif key == ord('h'):
                    # Show emoji list
                    self._show_emoji_list()
                elif key == ord('p'):
                    # Print performance info
                    self._print_performance_stats()
                
                frame_count += 1
                
                # Print FPS periodically
                if frame_count % fps_update_interval == 0:
                    current_time = time.time()
                    fps = fps_update_interval / (current_time - last_fps_time)
                    last_fps_time = current_time
                    expr_str = expressions[0] if expressions else "none"
                    print(f"📊 FPS: {fps:.1f} | Faces: {len(faces)} | Expression: {expr_str} | Detection: {detection_time*1000:.1f}ms")
                    
        except KeyboardInterrupt:
            print("\n⏹️  Detection stopped by user")
        finally:
            self.cleanup()
    
    def _add_info_overlay(self, frame, faces, expressions, detection_time):
        """Add information overlay to frame."""
        if len(self.frame_times) > 10:
            avg_frame_time = sum(self.frame_times[-10:]) / 10
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            # Current expression and emoji
            current_expr = expressions[0] if expressions else "none"
            if self.auto_mode and expressions:
                emoji_idx = self.select_emoji_for_expression(current_expr)
            else:
                emoji_idx = self.current_emoji_idx
            
            current_emoji = self.emoji_names[emoji_idx] if emoji_idx < len(self.emoji_names) else "none"
            mode = "AUTO" if self.auto_mode else "MANUAL"
            
            # Info text
            info_text = [
                f"FPS: {fps:.1f}",
                f"Faces: {len(faces)}",
                f"Expression: {current_expr}",
                f"Mode: {mode}",
                f"Emoji: {current_emoji}"
            ]
            
            y_offset = 30
            for i, text in enumerate(info_text):
                color = (0, 255, 0) if self.auto_mode else (0, 255, 255)
                cv2.putText(frame, text, (10, y_offset + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _show_emoji_list(self):
        """Show available emoji list."""
        print("\n🎭 Available Emojis:")
        expressions = ['Neutral', 'Smile', 'Open Mouth', 'Surprise']
        for i, name in enumerate(self.emoji_names, 1):
            expr = expressions[i-1] if i-1 < len(expressions) else f"Custom {i}"
            marker = "👉" if i-1 == self.current_emoji_idx else "  "
            print(f"{marker} {i}. {name} ({expr})")
        print()
    
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
        print(f"  Loaded emoji assets: {len(self.emojis)}")
        print(f"  Mode: {'AUTO' if self.auto_mode else 'MANUAL'}")
        print(f"  MediaPipe enabled: {MEDIAPIPE_AVAILABLE}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🧹 Cleanup completed")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Auto Face Emoji Detection Based on Expressions")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--assets-dir", default="Animoji", help="Directory containing emoji PNG files")
    parser.add_argument("--scale", type=float, default=2.1, help="Emoji scale factor")
    
    args = parser.parse_args()
    
    if not MEDIAPIPE_AVAILABLE:
        print("⚠️  For best results, install MediaPipe: pip install mediapipe")
        print("   Without MediaPipe, expression detection is not available.")
    
    # Create and run detector
    detector = AutoFaceEmojiDetector(camera_id=args.camera, assets_dir=args.assets_dir, scale_factor=args.scale)
    
    try:
        detector.setup()
        detector.run()
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        detector.cleanup()


if __name__ == "__main__":
    main() 