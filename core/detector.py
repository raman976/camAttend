import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis


class FaceDetector:
    def __init__(self, device=None):
        # Initialize RetinaFace detector from InsightFace
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        
    def detect_faces(self, image):
        # RetinaFace expects RGB image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb_image)
        
        if len(faces) == 0:
            return [], [], []
        
        # Extract boxes, confidence scores, and landmarks
        boxes = []
        probs = []
        landmarks = []
        
        for face in faces:
            # Bounding box: [x1, y1, x2, y2]
            bbox = face.bbox.astype(np.float32)
            boxes.append(bbox)
            
            # Detection confidence score
            probs.append(face.det_score)
            
            # Facial landmarks: 5 points (left_eye, right_eye, nose, left_mouth, right_mouth)
            kps = face.kps.astype(np.float32)
            landmarks.append(kps)
        
        return np.array(boxes), np.array(probs), np.array(landmarks)