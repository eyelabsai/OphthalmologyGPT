import markdown
from markupsafe import Markup
from flask import Flask, render_template, request, session
from openai import OpenAI
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from search_pubmed import search_pubmed
from semantic import semantic_search, tfidf_search

app = Flask(__name__)
app.secret_key = "my super secret key"  # change for production

# Set up OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="OPEN API KEY"
)

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

    manual_matches = []

    if request.method == "POST":
        user_input = request.form["query"]
        session["conversation"].append({"role": "user", "content": user_input})

        # If first few questions, fetch PubMed
        if len(session["conversation"]) <= 5:
            modified_query = user_input
            if not any(k in user_input for k in ["ocular", "ophthalmology", "eye"]):
                modified_query = "ocular " + user_input

            articles = search_pubmed(modified_query, last_n_years = 20, max_results=5)
            abstracts = [a['abstract'] for a in articles if a['abstract']]

            session["references"] = articles  # store references separately

            if abstracts:
                joined_abstracts = "\n\n".join(abstracts)
                keywords = extract_keywords(abstracts)
                session["conversation"].append({
                    "role": "system",
                    "content": f"Relevant PubMed articles found with abstracts: {', '.join(joined_abstracts)}"
                })
            else:
                session["conversation"].append({
                    "role": "system",
                    "content": "No relevant PubMed articles found."
                })
        #Wills Eye Manual
        manual_matches = semantic_search(user_input, top_k=3)
        if not manual_matches or all(m["score"] < 0.01 for m in manual_matches):
            manual_matches = tfidf_search(user_input, top_k=3)

        # Get LLM response
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",
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

        return render_template("chat.html", conversation=conversation_rendered, references=session["references"], manual_results=manual_matches)

    return render_template("index.html")



if __name__ == "__main__":
    app.run(debug=True)
