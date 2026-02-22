import os
import cv2

from core.detector import FaceDetector
from core.embedder import FaceEmbedder


class Enroller:

    def __init__(self, matcher):
        self.detector = FaceDetector()
        self.embedder = FaceEmbedder()
        self.matcher = matcher

    def enroll_dataset(self, folder):

        for file in os.listdir(folder):

            if file.startswith("."):
                continue

            path = os.path.join(folder, file)

            student_id = os.path.splitext(file)[0]

            image = cv2.imread(path)

            if image is None:
                print("Skipping:", file)
                continue

            boxes, probs, landmarks = self.detector.detect_faces(image)

            if len(boxes) == 0:
                print("No face in:", file)
                continue

            # Use first detected face with its landmark for proper alignment
            emb = self.embedder.get_embedding(image, bbox=boxes[0], landmark=landmarks[0])

            if emb is None:
                print("Embedding failed:", file)
                continue

            self.matcher.add_embedding(emb, student_id)

            print("Enrolled:", student_id)