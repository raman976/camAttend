import cv2
import torch
from facenet_pytorch import MTCNN

class FaceDetector:
    def __init__(self,device=None):
        if device is None:
            device="cuda" if torch.cuda.is_available() else "cpu"
        self.device=device
        self.mtcnn=MTCNN(
            keep_all=True,
            device=self.device
        )
    def detect_faces(self,image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes, probs = self.mtcnn.detect(rgb_image)

        faces = []

        if boxes is None:
            return faces, []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            face = image[y1:y2, x1:x2]
            faces.append(face)

        return faces, boxes