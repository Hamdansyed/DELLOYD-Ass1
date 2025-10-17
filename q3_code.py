import cv2
import mediapipe as mp

class FaceFeatureDetector:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Initialize models
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.7
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")

        print("Face Feature Detection Started")
        print("Press ESC to exit")

    def add_simple_ui(self, frame, face_count):
        """Add clean, minimal UI"""
        h, w = frame.shape[:2]
        
        # Simple header
        cv2.rectangle(frame, (0, 0), (w, 40), (50, 50, 50), -1)
        cv2.putText(frame, "Face Feature Detection", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Face count
        cv2.putText(frame, f"Faces: {face_count}", (w - 100, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Simple footer with instructions
        cv2.rectangle(frame, (0, h - 25), (w, h), (50, 50, 50), -1)
        cv2.putText(frame, "ESC: Exit", (10, h - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def process_frame(self, frame):
        """Process frame for face and feature detection"""
        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_count = 0
        
        # Face detection
        face_results = self.face_detection.process(rgb)
        if face_results.detections:
            for detection in face_results.detections:
                # Draw face bounding box
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                bw = int(bboxC.width * w)
                bh = int(bboxC.height * h)
                
                # Clean face rectangle
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 255, 255), 2)
                face_count += 1

        # Feature detection
        mesh_results = self.face_mesh.process(rgb)
        if mesh_results.multi_face_landmarks:
            for landmarks in mesh_results.multi_face_landmarks:
                h, w, _ = frame.shape

                # Nose tip - small green dot
                nose_tip = landmarks.landmark[1]
                cv2.circle(frame, 
                          (int(nose_tip.x * w), int(nose_tip.y * h)), 
                          3, (0, 255, 0), -1)

                # Precise eye points - very small blue dots
                left_eye_center = landmarks.landmark[468]  # Left iris center
                right_eye_center = landmarks.landmark[473]  # Right iris center
                
                # Small dots covering only the iris
                cv2.circle(frame, 
                          (int(left_eye_center.x * w), int(left_eye_center.y * h)), 
                          2, (255, 0, 0), -1)
                cv2.circle(frame, 
                          (int(right_eye_center.x * w), int(right_eye_center.y * h)), 
                          2, (255, 0, 0), -1)
                
                # Additional iris points for better coverage
                left_iris_points = [469, 470, 471, 472]
                right_iris_points = [474, 475, 476, 477]
                
                for point_idx in left_iris_points:
                    point = landmarks.landmark[point_idx]
                    cv2.circle(frame, 
                              (int(point.x * w), int(point.y * h)), 
                              1, (200, 0, 0), -1)
                
                for point_idx in right_iris_points:
                    point = landmarks.landmark[point_idx]
                    cv2.circle(frame, 
                              (int(point.x * w), int(point.y * h)), 
                              1, (200, 0, 0), -1)

        return frame, face_count

    def run(self):
        """Main application loop"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Process frame
                processed_frame, face_count = self.process_frame(frame)
                
                # Add clean UI
                self.add_simple_ui(processed_frame, face_count)

                # Display
                cv2.imshow("Face Features", processed_frame)

                # ESC to exit
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed")

if __name__ == "__main__":
    detector = FaceFeatureDetector()
    detector.run()