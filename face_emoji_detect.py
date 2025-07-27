#!/usr/bin/env python3
"""
Face Detection with Emoji Overlay
Real-time face detection with customizable emoji overlays

Controls:
- Press 'q' to quit
- Press '1-9' to change emoji
- Press 's' to save current frame
- Press 'r' to reset performance stats
"""

import time
import cv2
import numpy as np
import os
from urllib.request import urlopen
import argparse


class FaceEmojiDetector:
    def __init__(self, camera_id=0, emoji_size=100):
        """
        Initialize face detector with emoji overlay.
        
        Args:
            camera_id (int): Camera device ID
            emoji_size (int): Size of emoji overlay in pixels
        """
        self.camera_id = camera_id
        self.emoji_size = emoji_size
        self.cap = None
        self.face_cascade = None
        self.current_emoji_idx = 0
        self.emojis = []
        self.emoji_names = [
            "😀", "😎", "🤣", "😍", "🤔", 
            "😴", "🤯", "🥳", "😈"
        ]
        
        # Performance tracking
        self.frame_times = []
        self.detection_times = []
        
    def setup(self):
        """Setup face detector and camera."""
        print("🚀 Setting up Face Emoji Detector")
        
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
        
        # Create emoji overlays
        self._create_emojis()
        
    def _create_emojis(self):
        """Create emoji overlays as OpenCV images."""
        print("🎭 Creating emoji overlays...")
        
        # Create simple emoji-like shapes using OpenCV
        emoji_configs = [
            # Happy face 😀
            {"base_color": (0, 255, 255), "eye_color": (0, 0, 0), "mouth": "smile"},
            # Cool face 😎  
            {"base_color": (0, 255, 255), "eye_color": (50, 50, 50), "mouth": "smile", "sunglasses": True},
            # Laughing 🤣
            {"base_color": (0, 255, 255), "eye_color": (0, 0, 0), "mouth": "laugh"},
            # Heart eyes 😍
            {"base_color": (0, 255, 255), "eye_color": (0, 0, 255), "mouth": "smile", "heart_eyes": True},
            # Thinking 🤔
            {"base_color": (0, 255, 255), "eye_color": (0, 0, 0), "mouth": "neutral"},
            # Sleeping 😴
            {"base_color": (200, 200, 200), "eye_color": (0, 0, 0), "mouth": "neutral", "closed_eyes": True},
            # Mind blown 🤯
            {"base_color": (0, 255, 255), "eye_color": (255, 255, 255), "mouth": "surprise"},
            # Party 🥳
            {"base_color": (0, 255, 255), "eye_color": (0, 0, 0), "mouth": "smile", "party": True},
            # Devil 😈
            {"base_color": (0, 0, 255), "eye_color": (255, 255, 255), "mouth": "evil"}
        ]
        
        for config in emoji_configs:
            emoji = self._create_emoji_image(config)
            self.emojis.append(emoji)
        
        print(f"✅ Created {len(self.emojis)} emoji overlays")
        
    def _create_emoji_image(self, config):
        """Create an emoji image based on configuration."""
        size = self.emoji_size
        # Create transparent image (BGRA)
        emoji = np.zeros((size, size, 4), dtype=np.uint8)
        
        center = (size // 2, size // 2)
        radius = size // 2 - 10
        
        # Draw face (circle)
        cv2.circle(emoji, center, radius, (*config["base_color"], 255), -1)
        cv2.circle(emoji, center, radius, (0, 0, 0, 255), 3)
        
        # Draw eyes
        eye_y = center[1] - radius // 3
        eye_radius = radius // 8
        left_eye = (center[0] - radius//3, eye_y)
        right_eye = (center[0] + radius//3, eye_y)
        
        if config.get("closed_eyes"):
            # Closed eyes (lines)
            cv2.line(emoji, (left_eye[0]-eye_radius, left_eye[1]), 
                    (left_eye[0]+eye_radius, left_eye[1]), (0, 0, 0, 255), 3)
            cv2.line(emoji, (right_eye[0]-eye_radius, right_eye[1]), 
                    (right_eye[0]+eye_radius, right_eye[1]), (0, 0, 0, 255), 3)
        elif config.get("heart_eyes"):
            # Heart eyes
            self._draw_heart(emoji, left_eye, eye_radius)
            self._draw_heart(emoji, right_eye, eye_radius)
        else:
            # Normal eyes
            cv2.circle(emoji, left_eye, eye_radius, (*config["eye_color"], 255), -1)
            cv2.circle(emoji, right_eye, eye_radius, (*config["eye_color"], 255), -1)
        
        # Draw sunglasses if specified
        if config.get("sunglasses"):
            glass_rect1 = (left_eye[0]-eye_radius-5, left_eye[1]-eye_radius-5, 
                          2*eye_radius+10, 2*eye_radius+10)
            glass_rect2 = (right_eye[0]-eye_radius-5, right_eye[1]-eye_radius-5, 
                          2*eye_radius+10, 2*eye_radius+10)
            cv2.rectangle(emoji, glass_rect1[:2], 
                         (glass_rect1[0]+glass_rect1[2], glass_rect1[1]+glass_rect1[3]), 
                         (0, 0, 0, 255), -1)
            cv2.rectangle(emoji, glass_rect2[:2], 
                         (glass_rect2[0]+glass_rect2[2], glass_rect2[1]+glass_rect2[3]), 
                         (0, 0, 0, 255), -1)
        
        # Draw mouth
        mouth_y = center[1] + radius // 3
        mouth_center = (center[0], mouth_y)
        
        if config["mouth"] == "smile":
            # Smile
            cv2.ellipse(emoji, mouth_center, (radius//3, radius//5), 0, 0, 180, (0, 0, 0, 255), 3)
        elif config["mouth"] == "laugh":
            # Big laugh
            cv2.ellipse(emoji, mouth_center, (radius//2, radius//3), 0, 0, 180, (0, 0, 0, 255), -1)
        elif config["mouth"] == "surprise":
            # Surprised (O shape)
            cv2.circle(emoji, mouth_center, radius//6, (0, 0, 0, 255), -1)
        elif config["mouth"] == "evil":
            # Evil grin
            cv2.ellipse(emoji, mouth_center, (radius//3, radius//8), 0, 0, 180, (0, 0, 0, 255), 3)
        
        # Add party hat if specified
        if config.get("party"):
            hat_points = np.array([
                [center[0], center[1] - radius - 20],
                [center[0] - 15, center[1] - radius + 5],
                [center[0] + 15, center[1] - radius + 5]
            ], np.int32)
            cv2.fillPoly(emoji, [hat_points], (255, 0, 255, 255))
        
        return emoji
    
    def _draw_heart(self, img, center, size):
        """Draw a heart shape."""
        # Simple heart approximation using circles and triangle
        heart_color = (0, 0, 255, 255)  # Red heart
        
        # Two circles for top of heart
        cv2.circle(img, (center[0]-size//3, center[1]-size//4), size//3, heart_color, -1)
        cv2.circle(img, (center[0]+size//3, center[1]-size//4), size//3, heart_color, -1)
        
        # Triangle for bottom of heart
        triangle_points = np.array([
            [center[0]-size//2, center[1]],
            [center[0]+size//2, center[1]],
            [center[0], center[1]+size//2]
        ], np.int32)
        cv2.fillPoly(img, [triangle_points], heart_color)
    
    def detect_faces(self, frame):
        """Detect faces in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def overlay_emoji(self, frame, faces):
        """Overlay emoji on detected faces."""
        if len(self.emojis) == 0:
            return frame
            
        emoji = self.emojis[self.current_emoji_idx]
        
        for (x, y, w, h) in faces:
            # Calculate emoji position and size
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            
            # Scale emoji to face size
            emoji_scale = max(w, h) / self.emoji_size
            scaled_size = int(self.emoji_size * emoji_scale * 1.2)  # Make it slightly bigger
            
            # Resize emoji
            emoji_resized = cv2.resize(emoji, (scaled_size, scaled_size))
            
            # Calculate overlay position
            overlay_x = face_center_x - scaled_size // 2
            overlay_y = face_center_y - scaled_size // 2
            
            # Ensure overlay is within frame bounds
            overlay_x = max(0, min(overlay_x, frame.shape[1] - scaled_size))
            overlay_y = max(0, min(overlay_y, frame.shape[0] - scaled_size))
            
            # Apply emoji overlay with alpha blending
            self._apply_emoji_overlay(frame, emoji_resized, overlay_x, overlay_y)
        
        return frame
    
    def _apply_emoji_overlay(self, frame, emoji, x, y):
        """Apply emoji overlay with alpha blending."""
        h, w = emoji.shape[:2]
        
        # Ensure overlay fits within frame
        if x + w > frame.shape[1]:
            w = frame.shape[1] - x
            emoji = emoji[:, :w]
        if y + h > frame.shape[0]:
            h = frame.shape[0] - y
            emoji = emoji[:h, :]
        
        if w <= 0 or h <= 0:
            return
        
        # Extract alpha channel
        if emoji.shape[2] == 4:
            emoji_bgr = emoji[:, :, :3]
            alpha = emoji[:, :, 3] / 255.0
        else:
            emoji_bgr = emoji
            alpha = np.ones((h, w), dtype=float)
        
        # Get region of interest
        roi = frame[y:y+h, x:x+w]
        
        # Alpha blending
        for c in range(3):
            roi[:, :, c] = alpha * emoji_bgr[:, :, c] + (1 - alpha) * roi[:, :, c]
    
    def run(self):
        """Run real-time face detection with emoji overlay."""
        print("\n🎭 Starting Face Emoji Detection...")
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press '1-9' to change emoji")
        print("  - Press 's' to save current frame")
        print("  - Press 'r' to reset performance stats")
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
                cv2.imshow('Face Emoji Detection - Press q to quit', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key >= ord('1') and key <= ord('9'):
                    # Change emoji
                    emoji_num = key - ord('1')
                    if emoji_num < len(self.emojis):
                        self.current_emoji_idx = emoji_num
                        print(f"🎭 Changed to emoji {emoji_num + 1}: {self.emoji_names[emoji_num]}")
                elif key == ord('s'):
                    # Save current frame
                    save_path = f"face_emoji_detection_{save_count:04d}.jpg"
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
                f"Emoji: {self.emoji_names[self.current_emoji_idx]} (Press 1-9)"
            ]
            
            y_offset = 30
            for i, text in enumerate(info_text):
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
        print(f"  Current emoji: {self.emoji_names[self.current_emoji_idx]}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🧹 Cleanup completed")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Face Detection with Emoji Overlay")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--emoji-size", type=int, default=100, help="Base emoji size in pixels")
    
    args = parser.parse_args()
    
    # Create and run detector
    detector = FaceEmojiDetector(camera_id=args.camera, emoji_size=args.emoji_size)
    
    try:
        detector.setup()
        detector.run()
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        detector.cleanup()


if __name__ == "__main__":
    main() 