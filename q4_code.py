import cv2
import os
import time
from datetime import datetime
import numpy as np

class ModernFaceBlur:
    def __init__(self):
        # Try different camera indices
        self.cap = None
        for i in range(3):
            self.cap = cv2.VideoCapture(i)
            if self.cap.isOpened():
                ret, test_frame = self.cap.read()
                if ret:
                    print(f"✅ Camera found at index {i}")
                    break
            self.cap.release()
        
        if not self.cap or not self.cap.isOpened():
            print("❌ No camera found on indices 0-2")
            raise Exception("No camera detected")

        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.face_cascade.empty():
            raise Exception("Could not load face detection model")

        # Get camera resolution
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📷 Camera initialized: {self.width}x{self.height}")

        # Settings
        self.blur_strength = 50
        self.blur_enabled = True
        self.recording = False
        self.out = None
        
        # Stats
        self.fps = 0
        self.prev_time = 0
        self.face_count = 0
        self.recording_start = 0

        # Output folder
        os.makedirs("videos", exist_ok=True)
        
        print("🚀 Modern Face Blur App Started")
        print("Controls: Q=Quit, S=Record, B=Blur On/Off, +/-=Adjust Blur, C=Screenshot")

    def apply_face_blur(self, frame):
        """Detect and blur faces with modern visualization"""
        self.face_count = 0
        
        if not self.blur_enabled:
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Better detection
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        self.face_count = len(faces)

        for (x, y, w, h) in faces:
            # Expand blur area
            expand = 15
            x1 = max(0, x - expand)
            y1 = max(0, y - expand)
            x2 = min(frame.shape[1], x + w + expand)
            y2 = min(frame.shape[0], y + h + expand)

            # Apply blur
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size > 0:
                k = max(1, self.blur_strength // 2 * 2 + 1)  # Ensure odd number
                blurred_face = cv2.GaussianBlur(face_roi, (k, k), 0)
                frame[y1:y2, x1:x2] = blurred_face

            # Draw modern bounding box
            color = self.get_blur_color()
            thickness = 2
            
            # Main rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Corner accents
            corner_len = 12
            # Top-left
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
            # Top-right
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 3)
            # Bottom-left
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 3)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 3)
            # Bottom-right
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3)
            
            # Modern label
            label_bg = (x1, y1 - 35, 120, 25)
            cv2.rectangle(frame, (label_bg[0], label_bg[1]), 
                         (label_bg[0] + label_bg[2], label_bg[1] + label_bg[3]), 
                         color, -1)
            cv2.putText(frame, f"Blur: {self.blur_strength}%", 
                       (x1 + 8, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                       (255, 255, 255), 1)

        return frame

    def get_blur_color(self):
        """Get color based on blur strength"""
        if self.blur_strength < 33:
            return (0, 255, 0)    # Green - low
        elif self.blur_strength < 66:
            return (0, 255, 255)  # Yellow - medium
        else:
            return (0, 100, 255)  # Orange - high

    def update_fps(self):
        """Calculate smooth FPS"""
        current_time = time.time()
        if current_time - self.prev_time > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / (current_time - self.prev_time))
        self.prev_time = current_time

    def get_recording_time(self):
        """Get formatted recording time"""
        if not self.recording:
            return "00:00"
        elapsed = int(time.time() - self.recording_start)
        minutes = elapsed // 60
        seconds = elapsed % 60
        return f"{minutes:02d}:{seconds:02d}"

    def create_modern_ui(self, frame):
        """Create modern UI without complex gradients"""
        h, w = frame.shape[:2]
        
        # Modern header - simple dark overlay
        header_h = 70
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_h), (30, 30, 50), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # App title with modern font
        title = "FACE BLUR AI"
        cv2.putText(frame, title, (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        cv2.putText(frame, title, (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 150, 255), 2)
        
        # Modern stats panel
        stats_bg = (w - 220, 10, 210, 55)
        cv2.rectangle(frame, 
                     (stats_bg[0], stats_bg[1]), 
                     (stats_bg[0] + stats_bg[2], stats_bg[1] + stats_bg[3]), 
                     (40, 40, 60), -1)
        cv2.rectangle(frame, 
                     (stats_bg[0], stats_bg[1]), 
                     (stats_bg[0] + stats_bg[2], stats_bg[1] + stats_bg[3]), 
                     (100, 150, 255), 2)
        
        # Stats text
        stats = [
            f"Faces: {self.face_count}",
            f"FPS: {int(self.fps)}",
            f"Blur: {self.blur_strength}%"
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (stats_bg[0] + 10, stats_bg[1] + 20 + i * 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Modern recording indicator
        if self.recording:
            # Pulsing red dot
            pulse = int(abs(np.sin(time.time() * 6) * 5) + 5)
            cv2.circle(frame, (w - 240, 35), pulse + 2, (0, 0, 100), -1)
            cv2.circle(frame, (w - 240, 35), pulse, (0, 0, 255), -1)
            
            # Recording text with time
            rec_text = f"REC {self.get_recording_time()}"
            cv2.putText(frame, rec_text, (w - 220, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Status indicator at bottom
        status_bg = (10, h - 40, 200, 30)
        cv2.rectangle(frame, 
                     (status_bg[0], status_bg[1]), 
                     (status_bg[0] + status_bg[2], status_bg[1] + status_bg[3]), 
                     (40, 40, 60), -1)
        
        blur_status = "ON" if self.blur_enabled else "OFF"
        status_text = f"Blur: {blur_status} | {w}x{h}"
        cv2.putText(frame, status_text, (status_bg[0] + 10, status_bg[1] + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def toggle_recording(self, frame):
        """Start/stop recording with modern feedback"""
        if not self.recording:
            # Start recording
            filename = f"videos/recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            self.out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (self.width, self.height))
            self.recording = True
            self.recording_start = time.time()
            print(f"🔴 Recording started: {filename}")
        else:
            # Stop recording
            self.recording = False
            if self.out:
                self.out.release()
                self.out = None
            duration = time.time() - self.recording_start
            print(f"⏹️ Recording stopped ({duration:.1f}s)")

    def capture_screenshot(self, frame):
        """Take screenshot with modern feedback"""
        filename = f"videos/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")

    def adjust_blur(self, increase=True):
        """Adjust blur strength with smooth steps"""
        step = 5
        old_strength = self.blur_strength
        if increase:
            self.blur_strength = min(100, self.blur_strength + step)
        else:
            self.blur_strength = max(0, self.blur_strength - step)
        
        if old_strength != self.blur_strength:
            print(f"🔧 Blur strength: {self.blur_strength}%")

    def run(self):
        """Main application loop"""
        print("🔄 Starting camera feed...")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame from camera")
                    break

                self.update_fps()

                # Process frame
                processed = self.apply_face_blur(frame.copy())
                
                # Add modern UI
                self.create_modern_ui(processed)
                
                # Save if recording
                if self.recording and self.out:
                    self.out.write(processed)

                # Display in modern window
                cv2.imshow("Modern Face Blur AI", processed)

                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:  # Q or ESC
                    break
                elif key in [ord('s'), ord('r')]:  # S or R for record
                    self.toggle_recording(processed)
                elif key == ord('b'):
                    self.blur_enabled = not self.blur_enabled
                    status = "ON" if self.blur_enabled else "OFF"
                    print(f"🔧 Face blur: {status}")
                elif key in [ord('+'), ord('=')]:
                    self.adjust_blur(True)
                elif key in [ord('-'), ord('_')]:
                    self.adjust_blur(False)
                elif key == ord('c'):
                    self.capture_screenshot(processed)
                elif key == ord('d'):  # Debug info
                    print(f"📊 Debug - FPS: {int(self.fps)}, Faces: {self.face_count}, Frame: {frame.shape}")

        except KeyboardInterrupt:
            print("\n🛑 Application interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            # Cleanup
            self.cap.release()
            if self.out:
                self.out.release()
            cv2.destroyAllWindows()
            print("✅ Application closed successfully")

if __name__ == "__main__":
    try:
        app = ModernFaceBlur()
        app.run()
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        print("💡 Troubleshooting tips:")
        print("   - Check if webcam is connected")
        print("   - Make sure no other app is using the camera")
        print("   - Try running as administrator")
        print("   - Check camera drivers")