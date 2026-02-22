import cv2
from core.detector import FaceDetector

detector = FaceDetector()
image = cv2.imread("tests/test_images/seminar2.jpg")

print("Testing detection...")
boxes, probs, landmarks = detector.detect_faces(image)
print(f"Detected {len(boxes)} faces")
print(f"Probabilities: {probs}")
