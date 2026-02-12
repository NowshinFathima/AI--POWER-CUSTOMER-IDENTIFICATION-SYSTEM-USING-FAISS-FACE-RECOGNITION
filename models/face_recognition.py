from insightface.app import FaceAnalysis

face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

def get_embedding(image):
    faces = face_app.get(image)

    if len(faces) == 0:
        return None

    return faces[0].embedding
