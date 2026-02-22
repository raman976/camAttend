import cv2
from core.detector import FaceDetector
from core.embedder import FaceEmbedder
from core.matcher import FaceMatcher

detector = FaceDetector()
embedder = FaceEmbedder()
matcher = FaceMatcher()

image = cv2.imread("tests/test_images/seminar2.jpg")

faces, boxes = detector.detect_faces(image)

print("faces:", len(faces))

if len(faces) > 0:

    emb = embedder.get_embedding(faces[0])

    if emb is not None:

        print("embedding size:", len(emb))

        matcher.add_embedding(emb, "student_1")

        student, dist = matcher.match(emb)

        print("Matched:", student, "Distance:", dist)

    else:
        print("Embedding failed")