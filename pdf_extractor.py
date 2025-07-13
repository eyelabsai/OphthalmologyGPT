import os
import json
import numpy as np
import faiss
from nomic import embed
from nomic.cli import login
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import pickle

# ========================
# Settings & Paths
# ========================
login("nk-EQX3wlYi6_J8TGZ1ofiq6JpBaH0ufkYehoHbXyk5oFc")  # Must be called once

DATA_DIR = "./wills_eye_manual_embeddings/pdfminer_as_pdf_extractor/nomic_faiss_tfidf"
os.makedirs(DATA_DIR, exist_ok=True)

CHUNKS_PATH = f"{DATA_DIR}/chunks.json"
EMBED_PATH = f"{DATA_DIR}/embeddings.npy"
FAISS_PATH = f"{DATA_DIR}/faiss.index"
TFIDF_MATRIX_PATH = f"{DATA_DIR}/tfidf_matrix.npz"
VECTORIZER_PATH = f"{DATA_DIR}/vectorizer.pkl"

# ========================
# PDF Processing
# ========================
def extract_pdf_text(pdf_path):
    return extract_text(pdf_path)

def chunk_text(text, max_chars=2000, overlap=200):
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+max_chars]
        if chunk.strip():
            chunks.append(chunk.strip())
        i += max_chars - overlap
    return chunks

# ========================
# Load or Generate Chunks
# ========================
def load_chunks():
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_chunks(chunks):
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

# ========================
# Embeddings & FAISS
# ========================
def get_nomic_embeddings(texts, dimensionality=512):
    output = embed.text(
        texts=texts,
        model="nomic-embed-text-v1.5",
        task_type="search_document",
        dimensionality=dimensionality
    )
    return np.array(output['embeddings'])

def save_embeddings(embs):
    np.save(EMBED_PATH, embs)

def load_embeddings():
    return np.load(EMBED_PATH)

def build_faiss_index(embeddings):
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(normed.shape[1])
    index.add(normed.astype("float32"))
    return index

def save_faiss_index(index):
    faiss.write_index(index, FAISS_PATH)

def load_faiss_index():
    return faiss.read_index(FAISS_PATH)

def semantic_search(query, chunks, faiss_index, dim=512):
    query_emb = get_nomic_embeddings([query], dimensionality=dim)
    query_emb = query_emb / np.linalg.norm(query_emb)
    D, I = faiss_index.search(query_emb.astype("float32"), k=5)
    return [(chunks[i], float(D[0][idx])) for idx, i in enumerate(I[0])]

# ========================
# TF-IDF Fallback
# ========================
def build_tfidf_index(chunks):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(chunks)
    return tfidf_matrix, vectorizer

def save_tfidf(tfidf_matrix, vectorizer):
    sparse.save_npz(TFIDF_MATRIX_PATH, tfidf_matrix)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

def load_tfidf():
    tfidf_matrix = sparse.load_npz(TFIDF_MATRIX_PATH)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return tfidf_matrix, vectorizer

def tfidf_search(query, tfidf_matrix, chunks, vectorizer, top_k=5):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[-top_k:][::-1]
    return [(chunks[i], float(scores[i])) for i in top_indices]

# ========================
# Pipeline Entry Point
# ========================
def main():
    pdf_path = "Kalla Gervasio, Travis Peck - The Wills Eye Manual_ Office and Emergency Room Diagnosis and Treatment of Eye Disease (2021, LWW Wolters Kluwer) - libgen.li.pdf"
    dim = 512

    # Load or generate chunks
    chunks = load_chunks()
    if not chunks:
        print("Extracting & chunking PDF...")
        text = extract_pdf_text(pdf_path)
        chunks = chunk_text(text)
        save_chunks(chunks)

    # Load or compute embeddings + FAISS
    if os.path.exists(EMBED_PATH) and os.path.exists(FAISS_PATH):
        print("Loading cached FAISS index...")
        embeddings = load_embeddings()
        faiss_index = load_faiss_index()
    else:
        print("Computing embeddings and FAISS index...")
        embeddings = get_nomic_embeddings(chunks, dimensionality=dim)
        save_embeddings(embeddings)
        faiss_index = build_faiss_index(embeddings)
        save_faiss_index(faiss_index)

    # Load or build TF-IDF fallback
    if os.path.exists(TFIDF_MATRIX_PATH) and os.path.exists(VECTORIZER_PATH):
        tfidf_matrix, vectorizer = load_tfidf()
    else:
        tfidf_matrix, vectorizer = build_tfidf_index(chunks)
        save_tfidf(tfidf_matrix, vectorizer)

    # Run query
    query = "What are the symptoms of ocular rosacea?"
    print(f"\nQuery: {query}")

    results = semantic_search(query, chunks, faiss_index, dim=dim)
    if not results or all(score < 0.01 for _, score in results):
        print("Semantic search failed. Using TF-IDF fallback.")
        results = tfidf_search(query, tfidf_matrix, chunks, vectorizer)

    print("\nTop Results:\n")
    for passage, score in results:
        print(f"[Score: {score:.4f}]\n{passage}\n{'-'*50}")

if __name__ == "__main__":
    main()



