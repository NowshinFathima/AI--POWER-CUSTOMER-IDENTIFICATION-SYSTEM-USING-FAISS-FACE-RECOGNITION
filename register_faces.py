import os
import cv2
import numpy as np
import faiss
from insightface.app import FaceAnalysis

# Paths
IMAGE_DIR = "static/captured_faces"
FAISS_INDEX_PATH = "data/faiss.index"
EMBEDDINGS_PATH = "data/embeddings.npy"

# Create directories if missing
os.makedirs("data", exist_ok=True)

# Load face model
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

embeddings = []
names = []

for file in os.listdir(IMAGE_DIR):
    if file.lower().endswith((".jpg", ".png")):
        name = file.split("_")[0]
        img_path = os.path.join(IMAGE_DIR, file)

        img = cv2.imread(img_path)
        faces = face_app.get(img)

        if len(faces) == 0:
            print(f"No face found in {file}")
            continue

        emb = faces[0].embedding
        embeddings.append(emb)
        names.append(name)
        print(f"Registered: {file}")

embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
if len(embeddings) == 0:
    print("❌ No faces detected. Please add clear face images.")
    exit()

embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)


faiss.write_index(index, FAISS_INDEX_PATH)
np.save(EMBEDDINGS_PATH, np.array(names))

print("✅ Face registration completed")
