import os
import numpy as np
import faiss
from nomic import embed
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ========================
# Config & Auth
# ========================
import nomic

nomic.cli.login(token="nk-wxksOp2QRFbcpcXFIe1pazhBskDS82RtBgvcMIe_jME")
print("Logged In!!!!!")

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
# Nomic Embeddings
# ========================
def get_nomic_embeddings(texts, dimensionality=512):
    output = embed.text(
        texts=texts,
        model="nomic-embed-text-v1.5",
        task_type="search_document",
        dimensionality=dimensionality
    )
    embeddings = np.array(output['embeddings'])
    return embeddings

# ========================
# Build FAISS Index
# ========================
def build_faiss_index(embeddings):
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    dim = normed.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(normed.astype("float32"))
    return index, normed

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

def tfidf_search(query, tfidf_matrix, chunks, vectorizer, top_k=5):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[-top_k:][::-1]
    return [(chunks[i], float(scores[i])) for i in top_indices]

# ========================
# Pipeline Entry Point
# ========================
def main():
    pdf_path = "Kalla Gervasio, Travis Peck - The Wills Eye Manual_ Office and Emergency Room Diagnosis and Treatment of Eye Disease (2021, LWW Wolters Kluwer) - libgen.li.pdf"  # replace with your file
    print(f"Extracting text from: {pdf_path}")
    text = extract_pdf_text(pdf_path)

    print("Chunking text...")
    chunks = chunk_text(text)

    print("Generating Nomic embeddings...")
    embeddings = get_nomic_embeddings(chunks)

    print("Building FAISS index...")
    faiss_index, _ = build_faiss_index(embeddings)

    print("Building TF-IDF fallback...")
    tfidf_matrix, vectorizer = build_tfidf_index(chunks)

    # Query
    query = "What are the symptoms of ocular rosacea?"
    print(f"\nQuery: {query}")

    results = semantic_search(query, chunks, faiss_index)
    if not results or all(score < 0.01 for _, score in results):
        print("Semantic search weak or failed. Using TF-IDF fallback.")
        results = tfidf_search(query, tfidf_matrix, chunks, vectorizer)

    print("\nTop Results:\n")
    for passage, score in results:
        print(f"[Score: {score:.4f}]\n{passage}\n{'-'*50}")

if __name__ == "__main__":
    main()

