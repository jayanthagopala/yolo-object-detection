#!/usr/bin/env python3
"""
YOLO Object Detection with Face Emoji Overlay
Combines YOLO object detection with face-specific emoji overlays

This script detects people using YOLO and adds emoji overlays on face regions.
"""

import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO
import torch


class YOLOFaceEmojiDetector:
    def __init__(self, model_name="yolov8n.pt", camera_id=0, emoji_size=80):
        """
        Initialize YOLO detector with face emoji overlay.
        
        Args:
            model_name (str): YOLO model to use
            camera_id (int): Camera device ID
            emoji_size (int): Base emoji size in pixels
        """
        self.model_name = model_name
        self.camera_id = camera_id
        self.emoji_size = emoji_size
        self.model = None
        self.cap = None
        self.face_cascade = None
        self.current_emoji_idx = 0
        self.emojis = []
        self.emoji_names = ["😀", "😎", "🤣", "😍", "🤔", "😴", "🤯", "🥳", "😈"]
        
    def setup(self):
        """Setup YOLO model, face detector, and camera."""
        print("🚀 Setting up YOLO Face Emoji Detector")
        
        # Setup YOLO model
        if torch.backends.mps.is_available():
            print("✅ Using MPS acceleration")
        
        self.model = YOLO(self.model_name)
        print(f"✅ YOLO model loaded: {self.model_name}")
        
        # Setup face cascade for more precise face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        print("✅ Face detector loaded")
        
        # Setup camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Could not open camera {self.camera_id}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("✅ Camera setup complete")
        
        # Create emoji overlays
        self._create_simple_emojis()
        
    def _create_simple_emojis(self):
        """Create simple emoji overlays."""
        print("🎭 Creating emoji overlays...")
        
        # Simple emoji configurations
        emoji_configs = [
            {"color": (0, 255, 255), "mouth": "smile"},      # Happy 😀
            {"color": (0, 255, 255), "mouth": "cool"},       # Cool 😎
            {"color": (0, 255, 255), "mouth": "laugh"},      # Laugh 🤣
            {"color": (0, 255, 255), "mouth": "love"},       # Love 😍
            {"color": (100, 200, 255), "mouth": "think"},    # Think 🤔
            {"color": (150, 150, 150), "mouth": "sleep"},    # Sleep 😴
            {"color": (0, 255, 255), "mouth": "wow"},        # Wow 🤯
            {"color": (0, 255, 255), "mouth": "party"},      # Party 🥳
            {"color": (0, 100, 255), "mouth": "devil"}       # Devil 😈
        ]
        
        for config in emoji_configs:
            emoji = self._create_simple_emoji(config)
            self.emojis.append(emoji)
            
        print(f"✅ Created {len(self.emojis)} emoji overlays")
    
    def _create_simple_emoji(self, config):
        """Create a simple emoji image."""
        size = self.emoji_size
        emoji = np.zeros((size, size, 4), dtype=np.uint8)
        
        center = (size // 2, size // 2)
        radius = size // 2 - 5
        
        # Face circle
        cv2.circle(emoji, center, radius, (*config["color"], 255), -1)
        cv2.circle(emoji, center, radius, (0, 0, 0, 255), 2)
        
        # Eyes
        eye_y = center[1] - radius // 4
        left_eye = (center[0] - radius//3, eye_y)
        right_eye = (center[0] + radius//3, eye_y)
        eye_radius = radius // 10
        
        if config["mouth"] == "sleep":
            # Closed eyes
            cv2.line(emoji, (left_eye[0]-eye_radius, left_eye[1]), 
                    (left_eye[0]+eye_radius, left_eye[1]), (0, 0, 0, 255), 2)
            cv2.line(emoji, (right_eye[0]-eye_radius, right_eye[1]), 
                    (right_eye[0]+eye_radius, right_eye[1]), (0, 0, 0, 255), 2)
        elif config["mouth"] == "love":
            # Heart eyes
            self._draw_simple_heart(emoji, left_eye, eye_radius)
            self._draw_simple_heart(emoji, right_eye, eye_radius)
        else:
            # Normal eyes
            cv2.circle(emoji, left_eye, eye_radius, (0, 0, 0, 255), -1)
            cv2.circle(emoji, right_eye, eye_radius, (0, 0, 0, 255), -1)
        
        # Sunglasses for cool emoji
        if config["mouth"] == "cool":
            cv2.rectangle(emoji, (left_eye[0]-eye_radius-3, left_eye[1]-eye_radius-3),
                         (left_eye[0]+eye_radius+3, left_eye[1]+eye_radius+3), (0, 0, 0, 255), -1)
            cv2.rectangle(emoji, (right_eye[0]-eye_radius-3, right_eye[1]-eye_radius-3),
                         (right_eye[0]+eye_radius+3, right_eye[1]+eye_radius+3), (0, 0, 0, 255), -1)
        
        # Mouth
        mouth_y = center[1] + radius // 3
        mouth_center = (center[0], mouth_y)
        
        if config["mouth"] in ["smile", "cool", "party"]:
            cv2.ellipse(emoji, mouth_center, (radius//4, radius//6), 0, 0, 180, (0, 0, 0, 255), 2)
        elif config["mouth"] == "laugh":
            cv2.ellipse(emoji, mouth_center, (radius//3, radius//4), 0, 0, 180, (0, 0, 0, 255), -1)
        elif config["mouth"] == "wow":
            cv2.circle(emoji, mouth_center, radius//8, (0, 0, 0, 255), -1)
        elif config["mouth"] == "devil":
            cv2.ellipse(emoji, mouth_center, (radius//4, radius//8), 0, 0, 180, (0, 0, 0, 255), 2)
        
        return emoji
    
    def _draw_simple_heart(self, img, center, size):
        """Draw a simple heart."""
        heart_color = (0, 0, 255, 255)
        cv2.circle(img, (center[0]-size//2, center[1]-size//3), size//2, heart_color, -1)
        cv2.circle(img, (center[0]+size//2, center[1]-size//3), size//2, heart_color, -1)
        triangle = np.array([[center[0]-size, center[1]], [center[0]+size, center[1]], 
                           [center[0], center[1]+size]], np.int32)
        cv2.fillPoly(img, [triangle], heart_color)
    
    def detect_people_and_faces(self, frame):
        """Detect people using YOLO and faces using cascade."""
        # YOLO detection for people
        results = self.model(frame, verbose=False)
        people_boxes = []
        
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Class 0 is 'person' in COCO dataset
                    if class_id == 0 and confidence > 0.5:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        people_boxes.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
        
        # Face detection within people regions
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        all_faces = []
        
        if len(people_boxes) > 0:
            # Detect faces within person bounding boxes
            for (px, py, pw, ph) in people_boxes:
                # Extract person region
                person_roi = gray[py:py+ph, px:px+pw]
                
                # Detect faces in person region
                faces = self.face_cascade.detectMultiScale(
                    person_roi, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
                )
                
                # Convert face coordinates back to full frame
                for (fx, fy, fw, fh) in faces:
                    all_faces.append((px + fx, py + fy, fw, fh))
        else:
            # Fallback: detect faces in entire frame
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            all_faces = list(faces)
        
        return results, people_boxes, all_faces
    
    def overlay_emoji_on_faces(self, frame, faces):
        """Overlay emoji on detected faces."""
        if len(self.emojis) == 0 or len(faces) == 0:
            return frame
            
        emoji = self.emojis[self.current_emoji_idx]
        
        for (x, y, w, h) in faces:
            # Scale emoji to face size
            face_size = max(w, h)
            scaled_size = int(face_size * 1.3)  # Make emoji bigger than face
            
            # Resize emoji
            emoji_resized = cv2.resize(emoji, (scaled_size, scaled_size))
            
            # Calculate position (center on face)
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            overlay_x = face_center_x - scaled_size // 2
            overlay_y = face_center_y - scaled_size // 2
            
            # Ensure within bounds
            overlay_x = max(0, min(overlay_x, frame.shape[1] - scaled_size))
            overlay_y = max(0, min(overlay_y, frame.shape[0] - scaled_size))
            
            # Apply overlay
            self._apply_overlay(frame, emoji_resized, overlay_x, overlay_y)
        
        return frame
    
    def _apply_overlay(self, frame, overlay, x, y):
        """Apply overlay with alpha blending."""
        h, w = overlay.shape[:2]
        
        if x + w > frame.shape[1]:
            w = frame.shape[1] - x
            overlay = overlay[:, :w]
        if y + h > frame.shape[0]:
            h = frame.shape[0] - y
            overlay = overlay[:h, :]
        
        if w <= 0 or h <= 0:
            return
        
        # Alpha blending
        if overlay.shape[2] == 4:
            overlay_bgr = overlay[:, :, :3]
            alpha = overlay[:, :, 3] / 255.0
        else:
            overlay_bgr = overlay
            alpha = np.ones((h, w), dtype=float)
        
        roi = frame[y:y+h, x:x+w]
        
        for c in range(3):
            roi[:, :, c] = alpha * overlay_bgr[:, :, c] + (1 - alpha) * roi[:, :, c]
    
    def run(self):
        """Run real-time detection."""
        print("\n🎭 Starting YOLO Face Emoji Detection...")
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press '1-9' to change emoji")
        print("  - Press 's' to save frame")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Detect people and faces
                yolo_results, people_boxes, faces = self.detect_people_and_faces(frame)
                
                # Draw YOLO detections (people boxes)
                for (x, y, w, h) in people_boxes:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, 'Person', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Overlay emojis on faces
                frame = self.overlay_emoji_on_faces(frame, faces)
                
                # Add info
                info_text = [
                    f"People: {len(people_boxes)}",
                    f"Faces: {len(faces)}",
                    f"Emoji: {self.emoji_names[self.current_emoji_idx]} (Press 1-9)"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(frame, text, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                cv2.imshow('YOLO + Face Emoji Detection', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key >= ord('1') and key <= ord('9'):
                    emoji_num = key - ord('1')
                    if emoji_num < len(self.emojis):
                        self.current_emoji_idx = emoji_num
                        print(f"🎭 Emoji: {self.emoji_names[emoji_num]}")
                elif key == ord('s'):
                    save_path = f"yolo_face_emoji_{frame_count:04d}.jpg"
                    cv2.imwrite(save_path, frame)
                    print(f"💾 Saved: {save_path}")
                    frame_count += 1
                    
        except KeyboardInterrupt:
            print("\n⏹️  Stopped by user")
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="YOLO Face Emoji Detection")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--emoji-size", type=int, default=80, help="Emoji size")
    
    args = parser.parse_args()
    
    detector = YOLOFaceEmojiDetector(args.model, args.camera, args.emoji_size)
    
    try:
        detector.setup()
        detector.run()
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 