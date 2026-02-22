import cv2
import numpy as np
import insightface


class FaceEmbedder:

    def __init__(self):
        self.app = insightface.app.FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0)

        self.rec_model = self.app.models['recognition']

    def get_embedding(self, face_img):

        face_img = cv2.resize(face_img, (112, 112))
        

        embedding = self.rec_model.get_feat([face_img])
        
        return embedding[0]