import markdown
from markupsafe import Markup
from flask import Flask, render_template, request, session
from openai import OpenAI
import requests
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
app.secret_key = "my super secret key"  # change for production

# Set up OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

def fetch_pubmed_abstracts(query, max_results=5):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    email = "your_email@example.com"
    r = requests.get(f"{base}esearch.fcgi", params={
        "db": "pubmed", "term": query, "retmax": max_results, "retmode": "json", "email": email
    })
    pmids = r.json().get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return []

    r2 = requests.get(f"{base}efetch.fcgi", params={
        "db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "email": email
    })
    root = ET.fromstring(r2.content)
    out = []
    for art in root.findall(".//PubmedArticle"):
        langs = [l.text.lower() for l in art.findall(".//Language") if l.text]
        if "eng" not in langs:
            continue
        abstract = " ".join(a.text for a in art.findall(".//AbstractText") if a.text)
        title = art.findtext(".//ArticleTitle", default="No Title")
        pmid = art.findtext(".//PMID")
        doi = None
        for id in art.findall(".//ArticleId"):
            if id.attrib.get("IdType") == "doi":
                doi = id.text
        url = f"https://doi.org/{doi}" if doi else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
        out.append({"title": title, "abstract": abstract, "url": url})
    return out

def extract_keywords(texts, top_k=10):
    vec = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vec.fit_transform(texts)
    sums = np.asarray(X.sum(axis=0)).flatten()
    idx = sums.argsort()[-top_k:][::-1]
    return [vec.get_feature_names_out()[i] for i in idx]

@app.route("/", methods=["GET", "POST"])
def index():
    if "conversation" not in session:
        session["conversation"] = [
            {"role": "system", "content": "You are a helpful medical research assistant."}
        ]
    if "references" not in session:
        session["references"] = []

    if request.method == "POST":
        user_input = request.form["query"]
        session["conversation"].append({"role": "user", "content": user_input})

        # If first question, fetch PubMed
        if len(session["conversation"]) <= 2:
            modified_query = user_input
            if not any(k in user_input for k in ["ocular", "ophthalmology", "eye"]):
                modified_query = "ocular " + user_input

            articles = fetch_pubmed_abstracts(modified_query, max_results=5)
            abstracts = [a['abstract'] for a in articles if a['abstract']]

            session["references"] = articles  # store references separately

            if abstracts:
                joined_abstracts = "\n\n".join(abstracts)
                keywords = extract_keywords(abstracts)
                session["conversation"].append({
                    "role": "system",
                    "content": f"Relevant PubMed articles found with keywords: {', '.join(keywords)}"
                })
            else:
                session["conversation"].append({
                    "role": "system",
                    "content": "No relevant PubMed articles found."
                })

        # Get LLM response
        response = client.chat.completions.create(
            model="meta-llama/llama-3-70b-instruct",
            messages=session["conversation"]
        )
        assistant_reply = response.choices[0].message.content
        session["conversation"].append({"role": "assistant", "content": assistant_reply})

        # Render markdown into safe HTML
        conversation_rendered = []
        for msg in session["conversation"]:
            if msg["role"] == "assistant":
                content_html = Markup(markdown.markdown(msg["content"]))
                conversation_rendered.append({"role": msg["role"], "content": content_html})
            else:
                conversation_rendered.append(msg)

        return render_template("chat.html", conversation=conversation_rendered, references=session["references"])

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
