import requests
import xml.etree.ElementTree as ET
import json
from typing import List, Dict

# NCBI configuration
NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EMAIL = "shreyaskolte49@gmail.com"
TOOL = "MyPubmedFetcher"


def search_pubmed(query: str, retmax: int = 100) -> List[str]:
    url = f"{NCBI_BASE_URL}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "xml",
        "retmax": retmax,
        "tool": TOOL,
        "email": EMAIL
    }
    response = requests.get(url, params=params)
    root = ET.fromstring(response.text)
    return [id_elem.text for id_elem in root.findall(".//Id")]


def fetch_pubmed_details(pmids: List[str]) -> List[Dict]:
    url = f"{NCBI_BASE_URL}/efetch.fcgi"
    params = {
        "db": "pubmed",
        "retmode": "xml",
        "id": ",".join(pmids),
        "tool": TOOL,
        "email": EMAIL
    }
    response = requests.get(url, params=params)
    root = ET.fromstring(response.text)

    articles = []
    for article in root.findall(".//PubmedArticle"):
        try:
            pmid_elem = article.find(".//PMID")
            title_elem = article.find(".//ArticleTitle")
            abstract_elem = article.find(".//Abstract/AbstractText")
            journal_elem = article.find(".//Journal/Title")
            id_list = article.findall(".//ArticleId")

            title = title_elem.text if title_elem is not None else "N/A"
            abstract = abstract_elem.text if abstract_elem is not None else "N/A"
            journal = journal_elem.text if journal_elem is not None else "N/A"
            pmid = pmid_elem.text if pmid_elem is not None else "N/A"

            doi = None
            for id_elem in id_list:
                if id_elem.attrib.get("IdType") == "doi":
                    doi = id_elem.text
                    break

            authors = [
                f"{a.find('ForeName').text} {a.find('LastName').text}"
                for a in article.findall(".//Author")
                if a.find("ForeName") is not None and a.find("LastName") is not None
            ]

            articles.append({
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "authors": authors,
                "doi": doi,
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}",
                "pmid": pmid
            })
        except Exception as e:
            print(f"Skipping article due to error: {e}")
    return articles


def save_articles_to_json(articles: List[Dict], filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(articles)} articles to {filename}")


# Example usage
if __name__ == "__main__":
    query = "opthalmology trauma"
    pmids = search_pubmed(query, retmax=20)
    print(f"Found {len(pmids)} PMIDs.")

    if pmids:
        articles = fetch_pubmed_details(pmids)
        save_articles_to_json(articles, "pubmed_articles_with_links.json")
