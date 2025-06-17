from flask import Flask, render_template, request, jsonify
import requests
import xml.etree.ElementTree as ET
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)

EMAIL = "high.physicist2.71828@gmail.com"
TOOL = "PubMedUI"
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def clean_query(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    cleaned_tokens = [
        word.lower() for word in tokens
        if word.lower() not in stop_words and word.isalnum()
    ]
    return " ".join(cleaned_tokens)

def search_pubmed(query, retmax=10):
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "xml",
        "retmax": retmax,
        "tool": TOOL,
        "email": EMAIL
    }
    r = requests.get(f"{BASE_URL}/esearch.fcgi", params=search_params)
    root = ET.fromstring(r.text)
    pmids = [id_elem.text for id_elem in root.findall(".//Id")]
    return fetch_details(pmids)

def fetch_details(pmids):
    if not pmids:
        return []
    params = {
        "db": "pubmed",
        "retmode": "xml",
        "id": ",".join(pmids),
        "tool": TOOL,
        "email": EMAIL
    }
    r = requests.get(f"{BASE_URL}/efetch.fcgi", params=params)
    root = ET.fromstring(r.text)
    results = []

    for article in root.findall(".//PubmedArticle"):
        title = article.findtext(".//ArticleTitle", default="No Title")
        abstract = article.findtext(".//Abstract/AbstractText", default="No Abstract")
        journal = article.findtext(".//Journal/Title", default="No Journal")
        pmid = article.findtext(".//PMID", default="N/A")
        doi = next((id_elem.text for id_elem in article.findall(".//ArticleId")
                    if id_elem.attrib.get("IdType") == "doi"), None)
        authors = [
            f"{a.find('ForeName').text} {a.find('LastName').text}"
            for a in article.findall(".//Author")
            if a.find("ForeName") is not None and a.find("LastName") is not None
        ]
        results.append({
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "authors": authors,
            "doi": doi,
            "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
        })
    return results

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search")
def search():
    query = request.args.get("q", "")
    cleaned = clean_query(query)
    results = search_pubmed(query)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
