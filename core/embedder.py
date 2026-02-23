import numpy as np
import insightface
from insightface.utils import face_align


class FaceEmbedder:
    def __init__(self):
        self.app = insightface.app.FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0)
        self.rec_model = self.app.models['recognition']

    def get_embedding(self, image, bbox=None, landmark=None):
        if bbox is not None and landmark is not None:
            landmark = np.array(landmark, dtype=np.float32)
            aligned_face = face_align.norm_crop(image, landmark=landmark)
            embedding = self.rec_model.get_feat([aligned_face])[0]
            return embedding
        else:
            faces = self.app.get(image)
            if len(faces) == 0:
                return None
            return faces[0].embedding