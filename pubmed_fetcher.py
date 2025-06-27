# app.py
from flask import Flask, render_template, request
from llama_cpp import Llama
import requests, xml.etree.ElementTree as ET, numpy as np, nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

app = Flask(__name__)

# Initialize quantized Llama model
llm = Llama.from_pretrained(
    repo_id="Mungert/Meta-Llama-3-8B-Instruct-GGUF",
    filename="Meta-Llama-3-8B-Instruct-bf16-q4_k.gguf",
    n_threads=8,
    n_gpu_layers=0
)

# Summarization via Llama
def summarize_with_llama(text: str, max_tokens: int = 256) -> str:
    prompt = (
        "You are a medical research assistant. "
        "Summarize the following abstracts into a concise and professional paragraph:\n\n"
        f"{text}\n\nSummary:"
    )
    resp = llm(prompt, max_tokens=max_tokens, stop=["\n\n"])
    return resp['choices'][0]['text'].strip()

def extract_keywords(texts, top_k=10):
    vec = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vec.fit_transform(texts)
    sums = np.asarray(X.sum(axis=0)).flatten()
    idx = sums.argsort()[-top_k:][::-1]
    return [vec.get_feature_names_out()[i] for i in idx]

def rank_by_similarity(query, abstracts):
    # For semantic similarity, continue using BioBERT embeddings
    from transformers import AutoTokenizer, AutoModel
    import torch

    tok = AutoTokenizer.from_pretrained(
        "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    mdl = AutoModel.from_pretrained(
        "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

    def embed_list(lst):
        enc = tok(lst, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            out = mdl(**enc)
        return out.last_hidden_state.mean(dim=1)

    q_emb = embed_list([query])
    abs_emb = embed_list(abstracts)
    sims = cosine_similarity(q_emb, abs_emb)[0]
    return sorted(zip(abstracts, sims), key=lambda x: x[1], reverse=True)

def fetch_pubmed_abstracts(query, max_results=5):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    email = "your_email@example.com"
    r = requests.get(f"{base}esearch.fcgi", params={
        "db": "pubmed", "term": query, "retmax": max_results, "retmode": "json", "email": email})
    pmids = r.json().get("esearchresult", {}).get("idlist", [])
    if not pmids: return []

    r2 = requests.get(f"{base}efetch.fcgi", params={
        "db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "email": email})
    root = ET.fromstring(r2.content)
    out = []
    for art in root.findall(".//PubmedArticle"):
        langs = [l.text for l in art.findall(".//Language")]
        if "eng" not in [la.lower() for la in langs]:
            continue
        abstract = " ".join(a.text for a in art.findall(".//AbstractText") if a.text)
        title = art.findtext(".//ArticleTitle", default="No Title")
        pmid = art.findtext(".//PMID")
        doi = art.findtext(".//ArticleId[@IdType='doi']")
        url = f"https://doi.org/{doi}" if doi else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
        out.append({"title": title, "abstract": abstract, "url": url})
    return out

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["query"]
        if not any(k in query for k in ["ocular", "ophthalmology", "eye"]):
            query = "ocular " + query

        arts = fetch_pubmed_abstracts(query, max_results=5)
        if not arts:
            return render_template("results.html", query=query, summary="No results found.", keywords=[], articles=[])

        abstracts = [a['abstract'] for a in arts]
        ranked = rank_by_similarity(query, abstracts)
        sorted_abs = [r[0] for r in ranked]
        summary = summarize_with_llama(" ".join(sorted_abs))
        keywords = extract_keywords(sorted_abs)

        # Order articles by similarity
        abs_to_sim = dict(ranked)
        ranked_articles = sorted(arts, key=lambda a: abs_to_sim.get(a['abstract'], 0), reverse=True)

        return render_template(
            "results.html",
            query=query,
            summary=summary,
            keywords=keywords,
            articles=ranked_articles
        )
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
