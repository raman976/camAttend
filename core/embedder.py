import cv2
import numpy as np
import insightface
from insightface.utils import face_align


class FaceEmbedder:

    def __init__(self):
        self.app = insightface.app.FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0)
        # Get the recognition model for direct embedding extraction
        self.rec_model = self.app.models['recognition']

    def get_embedding(self, image, bbox=None, landmark=None):
        """
        Get face embedding from image.
        If bbox and landmark are provided, uses them directly.
        Otherwise, detects faces in the image.
        """
        if bbox is not None and landmark is not None:
            # Convert MTCNN landmarks (5 points) to numpy array
            landmark = np.array(landmark, dtype=np.float32)
            
            # Align the face using landmarks
            aligned_face = face_align.norm_crop(image, landmark=landmark)
            
            # Get embedding from aligned face
            embedding = self.rec_model.get_feat([aligned_face])[0]
            return embedding
        else:
            # Detect and process faces
            faces = self.app.get(image)
            if len(faces) == 0:
                return None
            # Return embedding of first face
            return faces[0].embedding
    
    def get_embeddings_batch(self, image):
        """
        Get embeddings for all faces detected in the image.
        Returns list of dicts with embedding and bbox.
        """
        faces = self.app.get(image)
        results = []
        for face in faces:
            results.append({
                "embedding": face.embedding,
                "bbox": face.bbox.astype(int).tolist()
            })
        return results