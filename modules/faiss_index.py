import faiss
import numpy as np
import os

INDEX_PATH = "data/faiss.index"

def create_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_index(index):
    faiss.write_index(index, INDEX_PATH)

def load_index():
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return None
