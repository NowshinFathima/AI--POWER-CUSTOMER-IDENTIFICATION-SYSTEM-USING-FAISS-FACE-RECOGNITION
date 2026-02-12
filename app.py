from flask import Flask, render_template, request
import numpy as np
import os
import cv2
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# =========================
# FOLDER SETUP
# =========================

if not os.path.exists("uploads"):
    os.makedirs("uploads")

if not os.path.exists("data"):
    os.makedirs("data")

DATA_FILE = "data/embeddings.npy"
NAMES_FILE = "data/names.npy"

# =========================
# LOAD EXISTING DATA
# =========================

if os.path.exists(DATA_FILE) and os.path.exists(NAMES_FILE):
    known_embeddings = list(np.load(DATA_FILE, allow_pickle=True))
    known_names = list(np.load(NAMES_FILE, allow_pickle=True))
else:
    known_embeddings = []
    known_names = []

# =========================
# INITIALIZE INSIGHTFACE
# =========================

face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)

# =========================
# EMBEDDING EXTRACTION
# =========================

def extract_embedding(image_path):
    img = cv2.imread(image_path)
    faces = face_app.get(img)

    if len(faces) == 0:
        return None

    embedding = faces[0].embedding
    embedding = embedding / np.linalg.norm(embedding)

    return embedding


# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/admin")
def admin():
    total_customers = len(known_names)
    return render_template("admin.html", 
                           customers=known_names,
                           total=total_customers)


@app.route("/identify", methods=["POST"])
def identify_customer():

    image = request.files["image"]
    filepath = os.path.join("uploads", image.filename)
    image.save(filepath)

    embedding = extract_embedding(filepath)

    if embedding is None:
        return render_template("index.html",
                               result="No face detected")

    # =========================
    # FIRST CUSTOMER
    # =========================
    if len(known_embeddings) == 0:
        known_embeddings.append(embedding)
        known_names.append("Customer_001")

        np.save(DATA_FILE, known_embeddings)
        np.save(NAMES_FILE, known_names)

        return render_template("index.html",
                               result="New Customer Registered (Customer_001)")

    # =========================
    # COMPARE WITH EXISTING
    # =========================
    known_array = np.array(known_embeddings)
    similarities = cosine_similarity([embedding], known_array)[0]

    best_score = float(max(similarities))
    best_index = int(np.argmax(similarities))

    if best_score > 0.75:
        result = f"Customer Recognized: {known_names[best_index]} (Score: {best_score:.2f})"
    else:
        new_id = f"Customer_{len(known_names)+1:03d}"
        known_embeddings.append(embedding)
        known_names.append(new_id)

        np.save(DATA_FILE, known_embeddings)
        np.save(NAMES_FILE, known_names)

        result = f"New Customer Registered ({new_id})"

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
