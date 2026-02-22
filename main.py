import cv2

from core.detector import FaceDetector
from core.embedder import FaceEmbedder
from core.matcher import FaceMatcher
from core.enroll import Enroller


matcher = FaceMatcher()

enroller = Enroller(matcher)

enroller.enroll_dataset("dataset/students")


detector = FaceDetector()
embedder = FaceEmbedder()

image = cv2.imread("tests/test_images/semianr.jpeg")

if image is None:
    print("Error: Could not load image")
    exit(1)

boxes, probs, landmarks = detector.detect_faces(image)

print("faces detected:", len(boxes))

for i, (box, landmark) in enumerate(zip(boxes, landmarks)):

    emb = embedder.get_embedding(image, bbox=box, landmark=landmark)
    
    if emb is None:
        continue
    
    student, score = matcher.match(emb)
    if score > 0.5:
        print(f"Face {i+1} - Present: {student}, score: {score:.3f}")
    else:
        print(f"Face {i+1} - Unknown face")
