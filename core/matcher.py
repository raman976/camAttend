import faiss
import numpy as np


class FaceMatcher:

    def __init__(self, dim=512):

        self.index = faiss.IndexFlatL2(dim)

        self.student_ids = []

    def add_embedding(self, embedding, student_id):

        embedding = np.array([embedding]).astype("float32")

        self.index.add(embedding)

        self.student_ids.append(student_id)

    def match(self, embedding):

        embedding = np.array([embedding]).astype("float32")

        distances, indices = self.index.search(embedding, k=1)

        idx = indices[0][0]

        if idx == -1:
            return None, None

        student_id = self.student_ids[idx]

        return student_id, distances[0][0]