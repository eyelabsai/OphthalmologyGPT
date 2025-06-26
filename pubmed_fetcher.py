from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import requests
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


app = Flask(__name__)

summarizer = pipeline("summarization", model="thisishadis/BioBart_on_pubmed")

embed_tokenizer = AutoTokenizer.from_pretrained("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
embed_model = AutoModel.from_pretrained("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

EMAIL = "high.physicist2.71828@gmail.com"
TOOL = "PubMedUI"
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def embed_texts(texts):
    max_len = 512
    inputs = embed_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len
    )
    with torch.no_grad():
        model_output = embed_model(**inputs)
    # Mean pooling
    return model_output.last_hidden_state.mean(dim=1)


def summarize_abstracts(abstracts):
    joined_text = " ".join(abstracts)[:2000]
    summary = summarizer(joined_text, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
    return summary

def extract_keywords(texts, top_k=10):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(texts)
    summed = np.asarray(X.sum(axis=0)).flatten()
    indices = summed.argsort()[-top_k:][::-1]
    return [vectorizer.get_feature_names_out()[i] for i in indices]

def rank_by_similarity(query, abstracts):
    query_embedding = embed_texts([query])
    abstract_embeddings = embed_texts(abstracts)
    similarities = cosine_similarity(query_embedding, abstract_embeddings)[0]
    return sorted(zip(abstracts, similarities), key=lambda x: x[1], reverse=True)

def fetch_pubmed_abstracts(query, max_results=5):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    email = "your_email@example.com"  # Replace with your actual email

    esearch_url = f"{base_url}esearch.fcgi?db=pubmed&term={query}&retmax={max_results}&retmode=json&email={email}"
    esearch_response = requests.get(esearch_url)
    esearch_data = esearch_response.json()
    id_list = esearch_data.get("esearchresult", {}).get("idlist", [])

    if not id_list:
        return []

    ids_str = ",".join(id_list)
    efetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={ids_str}&retmode=xml&email={email}"
    efetch_response = requests.get(efetch_url)

    root = ET.fromstring(efetch_response.content)
    results = []
    for article in root.findall(".//PubmedArticle"):
        languages = [lang.text.lower() for lang in article.findall(".//Language")]
        if not languages or "eng" not in languages:
            continue  # Skip non-English articles
        abstract_texts = article.findall(".//AbstractText")
        title_elem = article.find(".//ArticleTitle")
        pmid_elem = article.find(".//PMID")
        doi_elem = article.find(".//ArticleId[@IdType='doi']")

        abstract = " ".join(abstract_text.text for abstract_text in abstract_texts if abstract_text.text)
        title = title_elem.text if title_elem is not None else "No Title"
        pmid = pmid_elem.text if pmid_elem is not None else None
        doi = doi_elem.text if doi_elem is not None else None

        url = f"https://doi.org/{doi}" if doi else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
        results.append({
            "title": title,
            "abstract": abstract,
            "url": url
        })

    return results

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["query"]
        modified_query = query
        if 'ocular' not in query and 'ophthalmology' not in query and 'eye' not in query:
            modified_query = 'ocular ' + query
        articles = fetch_pubmed_abstracts(modified_query, max_results=5)

        if not articles:
            return render_template("results.html", query=query, summary="No articles found.", keywords=[], abstracts=[])

        abstracts = [article["abstract"] for article in articles if article["abstract"]]
        ranked = rank_by_similarity(query, abstracts)
        sorted_abstracts = [r[0] for r in ranked]

        summary = summarize_abstracts(sorted_abstracts)
        keywords = extract_keywords(sorted_abstracts)

        ranked_articles = sorted(articles, key=lambda a: next((r[1] for r in ranked if r[0] == a["abstract"]), 0), reverse=True)

        return render_template("results.html", query=query, summary=summary, keywords=keywords, articles=ranked_articles)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
