from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

summarizer = pipeline("summarization", model="thisishadis/BioBart_on_pubmed")

embed_tokenizer = AutoTokenizer.from_pretrained("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
embed_model = AutoModel.from_pretrained("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

def embed_texts(texts):
    inputs = embed_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        model_output = embed_model(**inputs)
    embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings

def summarize_abstracts(abstracts):
    joined_text = " ".join(abstracts)[:2000]
    summary = summarizer(joined_text, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
    return summary

def extract_keywords(texts, top_k=10):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(texts)
    summed = np.asarray(X.sum(axis=0)).flatten()
    indices = summed.argsort()[-top_k:][::-1]
    keywords = [vectorizer.get_feature_names_out()[i] for i in indices]
    return keywords

def rank_by_similarity(query, abstracts):
    query_embedding = embed_texts([query])
    abstract_embeddings = embed_texts(abstracts)
    similarities = cosine_similarity(query_embedding, abstract_embeddings)[0]
    ranked = sorted(zip(abstracts, similarities), key=lambda x: x[1], reverse=True)
    return ranked

# if __name__ == "__main__":
#     abstracts = [
#         "This study explores the use of AI in detecting lung cancer from imaging scans.",
#         "The effects of immunotherapy in advanced melanoma cases are evaluated.",
#         "A new vaccine for HPV shows promising results in clinical trials."
#     ]
#     user_query = "Recent advancements in cancer detection using artificial intelligence"
#
#     summary = summarize_abstracts(abstracts)
#     keywords = extract_keywords(abstracts)
#     ranked = rank_by_similarity(user_query, abstracts)

    print("\nSummary:\n", summary)
    print("\nKeywords:\n", keywords)
    print("\nRanked Abstracts:\n", [text for text, _ in ranked])