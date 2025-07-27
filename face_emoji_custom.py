#!/usr/bin/env python3
"""
Face Detection with Custom Emoji Overlay
Real-time face detection with high-quality custom emoji assets

This version supports loading your own emoji images for better quality overlays.
Place your emoji PNG files (with transparency) in the 'emoji_assets' folder.
"""

import time
import cv2
import numpy as np
import os
import argparse
from pathlib import Path


class CustomFaceEmojiDetector:
    def __init__(self, camera_id=0, emoji_size=100, assets_dir="emoji_assets", scale_factor=2.1):
        """
        Initialize face detector with custom emoji assets.
        
        Args:
            camera_id (int): Camera device ID
            emoji_size (int): Base emoji size in pixels
            assets_dir (str): Directory containing emoji PNG files
            scale_factor (float): How much larger emoji should be than face (1.0 = same size)
        """
        self.camera_id = camera_id
        self.emoji_size = emoji_size
        self.assets_dir = Path(assets_dir)
        self.scale_factor = scale_factor
        self.cap = None
        self.face_cascade = None
        self.current_emoji_idx = 0
        self.emojis = []
        self.emoji_names = []
        self.emoji_files = []
        
        # Performance tracking
        self.frame_times = []
        self.detection_times = []
        
    def setup(self):
        """Setup face detector and camera."""
        print("🚀 Setting up Custom Face Emoji Detector")
        
        # Load face cascade classifier
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise RuntimeError("❌ Could not load face cascade classifier")
        print("✅ Face detector loaded")
        
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
        """Load custom emoji assets from directory."""
        print(f"🎭 Loading emoji assets from {self.assets_dir}...")
        
        # Create assets directory if it doesn't exist
        self.assets_dir.mkdir(exist_ok=True)
        
        # Look for PNG files in assets directory
        emoji_files = sorted(list(self.assets_dir.glob("*.png")))
        
        if not emoji_files:
            print("⚠️  No PNG files found in emoji_assets directory!")
            print("📁 Creating sample emoji assets...")
            self._create_sample_assets()
            emoji_files = sorted(list(self.assets_dir.glob("*.png")))
        
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
        print("📝 Available emojis:")
        for i, name in enumerate(self.emoji_names, 1):
            print(f"  {i}. {name}")
    
    def _create_sample_assets(self):
        """Create sample emoji assets if none exist."""
        print("🎨 Creating sample emoji assets...")
        
        # Define sample emojis with better quality
        sample_emojis = [
            ("happy", (0, 255, 255), "smile"),
            ("cool", (0, 255, 255), "cool"),  
            ("laugh", (0, 255, 255), "laugh"),
            ("love", (255, 100, 255), "love"),
            ("think", (100, 200, 255), "think"),
            ("sleep", (150, 150, 150), "sleep"),
            ("wow", (0, 255, 255), "wow"),
            ("party", (255, 200, 0), "party"),
            ("devil", (0, 100, 255), "devil")
        ]
        
        for name, color, style in sample_emojis:
            emoji = self._create_high_quality_emoji(color, style)
            filename = self.assets_dir / f"{name}.png"
            cv2.imwrite(str(filename), emoji)
            print(f"✅ Created: {filename.name}")
    
    def _create_high_quality_emoji(self, base_color, style):
        """Create a higher quality emoji with better rendering."""
        size = 200  # Higher resolution for better quality
        emoji = np.zeros((size, size, 4), dtype=np.uint8)
        
        center = (size // 2, size // 2)
        radius = size // 2 - 20
        
        # Anti-aliased face circle with gradient
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        
        # Create gradient effect
        y, x = np.ogrid[:size, :size]
        center_y, center_x = center
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        gradient = 1 - (dist_from_center / radius) * 0.3
        gradient = np.clip(gradient, 0.7, 1.0)
        
        # Apply base color with gradient
        for c in range(3):
            emoji[:, :, c] = (mask / 255.0) * base_color[c] * gradient
        emoji[:, :, 3] = mask  # Alpha channel
        
        # Add face features with anti-aliasing
        self._add_face_features(emoji, center, radius, style)
        
        return emoji
    
    def _add_face_features(self, emoji, center, radius, style):
        """Add detailed face features."""
        # Eyes
        eye_y = center[1] - radius // 3
        eye_radius = radius // 8
        left_eye = (center[0] - radius//3, eye_y)
        right_eye = (center[0] + radius//3, eye_y)
        
        if style == "sleep":
            # Closed eyes with eyelashes
            for offset in range(-2, 3):
                cv2.line(emoji, (left_eye[0]-eye_radius, left_eye[1]+offset), 
                        (left_eye[0]+eye_radius, left_eye[1]+offset), (0, 0, 0, 255), 2)
                cv2.line(emoji, (right_eye[0]-eye_radius, right_eye[1]+offset), 
                        (right_eye[0]+eye_radius, right_eye[1]+offset), (0, 0, 0, 255), 2)
        elif style == "love":
            # Heart eyes with better shape
            self._draw_detailed_heart(emoji, left_eye, eye_radius)
            self._draw_detailed_heart(emoji, right_eye, eye_radius)
        else:
            # Normal eyes with pupils
            cv2.circle(emoji, left_eye, eye_radius, (255, 255, 255, 255), -1)
            cv2.circle(emoji, right_eye, eye_radius, (255, 255, 255, 255), -1)
            cv2.circle(emoji, left_eye, eye_radius//2, (0, 0, 0, 255), -1)
            cv2.circle(emoji, right_eye, eye_radius//2, (0, 0, 0, 255), -1)
        
        # Sunglasses for cool style
        if style == "cool":
            glass_thickness = 8
            cv2.rectangle(emoji, (left_eye[0]-eye_radius-5, left_eye[1]-eye_radius-5),
                         (left_eye[0]+eye_radius+5, left_eye[1]+eye_radius+5), (0, 0, 0, 255), glass_thickness)
            cv2.rectangle(emoji, (right_eye[0]-eye_radius-5, right_eye[1]-eye_radius-5),
                         (right_eye[0]+eye_radius+5, right_eye[1]+eye_radius+5), (0, 0, 0, 255), glass_thickness)
            # Bridge
            cv2.line(emoji, (left_eye[0]+eye_radius+5, left_eye[1]), 
                    (right_eye[0]-eye_radius-5, right_eye[1]), (0, 0, 0, 255), 4)
        
        # Mouth
        mouth_y = center[1] + radius // 3
        mouth_center = (center[0], mouth_y)
        
        if style in ["smile", "cool", "party"]:
            # Smiling mouth
            cv2.ellipse(emoji, mouth_center, (radius//3, radius//5), 0, 0, 180, (0, 0, 0, 255), 4)
        elif style == "laugh":
            # Big laughing mouth
            cv2.ellipse(emoji, mouth_center, (radius//2, radius//3), 0, 0, 180, (0, 0, 0, 255), -1)
            # Teeth
            cv2.ellipse(emoji, mouth_center, (radius//2-4, radius//3-4), 0, 0, 180, (255, 255, 255, 255), -1)
        elif style == "wow":
            # Surprised mouth
            cv2.circle(emoji, mouth_center, radius//6, (0, 0, 0, 255), -1)
        elif style == "devil":
            # Evil grin
            cv2.ellipse(emoji, mouth_center, (radius//3, radius//8), 0, 0, 180, (0, 0, 0, 255), 4)
        
        # Party hat
        if style == "party":
            hat_points = np.array([
                [center[0], center[1] - radius - 30],
                [center[0] - 20, center[1] - radius + 10],
                [center[0] + 20, center[1] - radius + 10]
            ], np.int32)
            cv2.fillPoly(emoji, [hat_points], (255, 0, 255, 255))
            # Hat decoration
            cv2.circle(emoji, (center[0], center[1] - radius - 25), 5, (255, 255, 0, 255), -1)
    
    def _draw_detailed_heart(self, img, center, size):
        """Draw a detailed heart shape."""
        heart_color = (0, 0, 255, 255)  # Red
        
        # More detailed heart shape
        scale = size / 10
        
        # Heart curves
        for angle in range(0, 360, 5):
            rad = np.radians(angle)
            # Heart equation: x = 16sin³(t), y = 13cos(t) - 5cos(2t) - 2cos(3t) - cos(4t)
            x = int(center[0] + scale * 16 * (np.sin(rad)**3))
            y = int(center[1] - scale * (13*np.cos(rad) - 5*np.cos(2*rad) - 2*np.cos(3*rad) - np.cos(4*rad)))
            
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                cv2.circle(img, (x, y), 2, heart_color, -1)
    
    def detect_faces(self, frame):
        """Detect faces in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with optimized parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def overlay_emoji(self, frame, faces):
        """Overlay custom emoji on detected faces."""
        if len(self.emojis) == 0 or len(faces) == 0:
            return frame
            
        emoji = self.emojis[self.current_emoji_idx]
        
        for (x, y, w, h) in faces:
            # Calculate emoji position and size
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            
            # Scale emoji to face size with better proportions
            face_size = max(w, h)
            scaled_size = int(face_size * self.scale_factor)  # Make emoji larger than face for better coverage
            
            # Resize emoji with high-quality interpolation
            emoji_resized = cv2.resize(emoji, (scaled_size, scaled_size), interpolation=cv2.INTER_CUBIC)
            
            # Calculate overlay position (center on face)
            overlay_x = face_center_x - scaled_size // 2
            overlay_y = face_center_y - scaled_size // 2
            
            # Ensure overlay is within frame bounds
            overlay_x = max(0, min(overlay_x, frame.shape[1] - scaled_size))
            overlay_y = max(0, min(overlay_y, frame.shape[0] - scaled_size))
            
            # Apply emoji overlay with smooth alpha blending
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
        
        # Smooth alpha blending with gamma correction
        alpha_smooth = np.power(alpha, 0.8)  # Gamma correction for smoother blending
        
        # Get region of interest
        roi = frame[y:y+h, x:x+w]
        
        # Apply blending
        for c in range(3):
            roi[:, :, c] = alpha_smooth * overlay_bgr[:, :, c] + (1 - alpha_smooth) * roi[:, :, c]
    
    def run(self):
        """Run real-time face detection with custom emoji overlay."""
        print("\n🎭 Starting Custom Face Emoji Detection...")
        print("Controls:")
        print("  - Press 'q' to quit")
        print(f"  - Press '1-{len(self.emojis)}' to change emoji")
        print("  - Press 's' to save current frame")
        print("  - Press 'r' to reset performance stats")
        print("  - Press 'p' to print performance info")
        print("  - Press 'h' to show emoji list")
        
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
                
                # Detect faces
                detection_start = time.time()
                faces = self.detect_faces(frame)
                detection_time = time.time() - detection_start
                
                # Overlay emojis on faces
                frame = self.overlay_emoji(frame, faces)
                
                # Add info overlay
                self._add_info_overlay(frame, faces, detection_time)
                
                # Calculate frame time
                frame_time = time.time() - frame_start
                self.frame_times.append(frame_time)
                self.detection_times.append(detection_time)
                
                # Keep only recent times
                if len(self.frame_times) > 100:
                    self.frame_times.pop(0)
                    self.detection_times.pop(0)
                
                # Display frame
                cv2.imshow('Custom Face Emoji Detection - Press q to quit', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key >= ord('1') and key <= ord('9'):
                    # Change emoji
                    emoji_num = key - ord('1')
                    if emoji_num < len(self.emojis):
                        self.current_emoji_idx = emoji_num
                        print(f"🎭 Changed to emoji: {self.emoji_names[emoji_num]}")
                elif key == ord('s'):
                    # Save current frame
                    save_path = f"custom_face_emoji_{save_count:04d}.jpg"
                    cv2.imwrite(save_path, frame)
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
                elif key == ord('h'):
                    # Show emoji list
                    self._show_emoji_list()
                
                frame_count += 1
                
                # Print FPS periodically
                if frame_count % fps_update_interval == 0:
                    current_time = time.time()
                    fps = fps_update_interval / (current_time - last_fps_time)
                    last_fps_time = current_time
                    print(f"📊 FPS: {fps:.1f} | Faces: {len(faces)} | Detection: {detection_time*1000:.1f}ms")
                    
        except KeyboardInterrupt:
            print("\n⏹️  Detection stopped by user")
        finally:
            self.cleanup()
    
    def _add_info_overlay(self, frame, faces, detection_time):
        """Add information overlay to frame."""
        if len(self.frame_times) > 10:
            avg_frame_time = sum(self.frame_times[-10:]) / 10
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            # Info text
            info_text = [
                f"FPS: {fps:.1f}",
                f"Faces: {len(faces)}",
                f"Detection: {detection_time*1000:.1f}ms",
                f"Emoji: {self.emoji_names[self.current_emoji_idx]} (Press 1-{len(self.emojis)})"
            ]
            
            y_offset = 30
            for i, text in enumerate(info_text):
                cv2.putText(frame, text, (10, y_offset + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def _show_emoji_list(self):
        """Show available emoji list."""
        print("\n🎭 Available Emojis:")
        for i, name in enumerate(self.emoji_names, 1):
            marker = "👉" if i-1 == self.current_emoji_idx else "  "
            print(f"{marker} {i}. {name}")
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
        print(f"  Average frame time: {avg_frame_time*1000:.2f}ms")
        print(f"  Total frames processed: {len(self.frame_times)}")
        print(f"  Current emoji: {self.emoji_names[self.current_emoji_idx]}")
        print(f"  Loaded emoji assets: {len(self.emojis)}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🧹 Cleanup completed")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Face Detection with Custom Emoji Assets")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--emoji-size", type=int, default=100, help="Base emoji size in pixels")
    parser.add_argument("--assets-dir", default="emoji_assets", help="Directory containing emoji PNG files")
    parser.add_argument("--scale", type=float, default=2.1, help="Emoji scale factor (how much larger than face, e.g., 1.5, 2.0)")
    
    args = parser.parse_args()
    
    # Create and run detector
    detector = CustomFaceEmojiDetector(camera_id=args.camera, emoji_size=args.emoji_size, assets_dir=args.assets_dir, scale_factor=args.scale)
    
    try:
        detector.setup()
        detector.run()
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        detector.cleanup()


if __name__ == "__main__":
    main() 