import requests
import xml.etree.ElementTree as ET
from datetime import datetime

def search_pubmed(
    query,
    reviews_only=False,
    clinical_trials=False,
    mesh_major_topic=False,
    last_n_years=None,
    max_results=10,
    email="your_email@example.com"
):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    advanced_query = f"({query}[majr])" if mesh_major_topic else f"({query}[Title/Abstract])"
    # advanced_query += " AND english[lang] AND hasabstract[text]"
    if reviews_only:
        advanced_query += " AND review[pt]"
    if clinical_trials:
        advanced_query += " AND clinicaltrial[pt]"
    if last_n_years:
        start_year = datetime.now().year - last_n_years
        advanced_query += f" AND (\"{start_year}\"[Date - Publication] : \"3000\"[Date - Publication])"

    esearch_url = (
        f"{base_url}esearch.fcgi?db=pubmed"
        f"&term={advanced_query}"
        f"&retmax={max_results}"
        f"&retmode=json"
        f"&sort=relevance"
        f"&email={email}"
    )
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
        abstract_texts = article.findall(".//AbstractText")
        title_elem = article.find(".//ArticleTitle")
        pmid_elem = article.find(".//PMID")
        doi_elem = article.find(".//ArticleId[@IdType='doi']")

        abstract = " ".join(at.text for at in abstract_texts if at.text)
        title = title_elem.text if title_elem is not None else "No Title"
        pmid = pmid_elem.text if pmid_elem is not None else None
        doi = doi_elem.text if doi_elem is not None else None
        url = f"https://doi.org/{doi}" if doi else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
        results.append({"title": title, "abstract": abstract, "url": url})
    return results
