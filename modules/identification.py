import numpy as np
from modules.faiss_index import load_index, create_index, save_index
from database.db import fetch_customers

THRESHOLD = 1.0   # tune later

def identify_face(query_embedding):
    customers = fetch_customers()

    if not customers:
        return "No customers registered"

    ids, names, embeddings = zip(*customers)
    emb_matrix = np.vstack(embeddings).astype("float32")

    index = load_index()
    if index is None:
        index = create_index(emb_matrix)
        save_index(index)

    D, I = index.search(np.array([query_embedding]), 1)

    if D[0][0] > THRESHOLD:
        return "Unknown Customer"

    return names[I[0][0]]
