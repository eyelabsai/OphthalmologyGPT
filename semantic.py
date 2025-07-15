# semantic.py
import os, json, numpy as np, faiss, pickle
from nomic import embed
from nomic.cli import login
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

login("your-nomic-api-key")  # call once

DATA_DIR = "./wills_eye_manual_embeddings/pdfminer_as_pdf_extractor/nomic_faiss_tfidf"
EMBED_DIM = 512

chunks = json.load(open(f"{DATA_DIR}/chunks.json", encoding="utf-8"))
embeddings = np.load(f"{DATA_DIR}/embeddings.npy")
faiss_index = faiss.read_index(f"{DATA_DIR}/faiss.index")
tfidf_matrix = sparse.load_npz(f"{DATA_DIR}/tfidf_matrix.npz")
vectorizer = pickle.load(open(f"{DATA_DIR}/vectorizer.pkl", "rb"))

def semantic_search(query, top_k=5):
    query_emb = embed.text(
        texts=[query],
        model="nomic-embed-text-v1.5",
        task_type="search_query",
        dimensionality=EMBED_DIM
    )['embeddings'][0]
    normed = np.array(query_emb) / np.linalg.norm(query_emb)
    D, I = faiss_index.search(np.array([normed]).astype("float32"), k=top_k)
    return [{"text": chunks[i], "score": float(D[0][rank])} for rank, i in enumerate(I[0])]

def tfidf_search(query, top_k=5):
    qvec = vectorizer.transform([query])
    scores = cosine_similarity(qvec, tfidf_matrix).flatten()
    indices = scores.argsort()[-top_k:][::-1]
    return [{"text": chunks[i], "score": float(scores[i])} for i in indices]
