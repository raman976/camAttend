import cv2
from core.detector import FaceDetector
from core.embedder import FaceEmbedder

detector = FaceDetector()
embedder = FaceEmbedder()

image = cv2.imread("tests/test_images/seminar2.jpg")

faces, boxes = detector.detect_faces(image)

print("faces:", len(faces))

if faces:
    emb = embedder.get_embedding(faces[0])
    print("embedding size:", len(emb))