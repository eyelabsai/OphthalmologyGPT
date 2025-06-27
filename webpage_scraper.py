import requests
from bs4 import BeautifulSoup
import json

def scrape_cureus_article(url: str) -> dict:
    res = requests.get(url)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")

    title = soup.find("h1").get_text(strip=True)
    authors = [a.strip() for a in soup.select(".authors")]

    sections = {}
    for heading in soup.find_all(["h2", "h3"]):
        sec_title = heading.get_text(strip=True).lower()
        if sec_title in ["abstract", "introduction", "methods", "conclusion", "discussion"]:
            # Gather all <p> until next heading
            content = []
            for sib in heading.find_next_siblings():
                if sib.name in ["h2", "h3"]:
                    break
                if sib.name == "p":
                    content.append(sib.get_text(strip=True))
            sections[sec_title] = "\n\n".join(content)

    return {
        "url": url,
        "title": title,
        "authors": authors,
        "sections": sections
    }

if __name__ == "__main__":
    url = "https://www.cureus.com/articles/341723-a-case-based-approach-to-the-management-of-corneal-melts-and-perforations-in-ocular-surface-disorders"
    article = scrape_cureus_article(url)
    print(json.dumps(article, indent=2))
