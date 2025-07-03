import json
import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

def extract_text_by_page(pdf_path):
    pages = []
    for i, page_layout in enumerate(extract_pages(pdf_path)):
        text_content = ""
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text_content += element.get_text()
        pages.append({"page": i+1, "text": text_content.strip()})
    return pages

def get_embedding(text):
    response = client.embeddings.create(
        model="deepseek/deepseek-r1:free",
        input=text
    )
    return response.data[0].embedding

def build_book_json(pdf_path, output_path):
    pages = extract_text_by_page(pdf_path)
    print(f"Extracted {len(pages)} pages. Generating embeddings...")

    book_data = []
    for page in pages:
        text = page["text"]
        if len(text) < 20:
            continue  # skip very short pages
        embedding = get_embedding(text)
        book_data.append({
            "page": page["page"],
            "text": text,
            "embedding": embedding
        })
        print(f"Page {page['page']} processed with embedding length {len(embedding)}")

    with open(output_path, "w") as f:
        json.dump(book_data, f, indent=2)
    print(f"\n✅ Finished. Saved to {output_path}")

if __name__ == "__main__":
    build_book_json("Kalla Gervasio, Travis Peck - The Wills Eye Manual_ Office and Emergency Room Diagnosis and Treatment of Eye Disease (2021, LWW Wolters Kluwer) - libgen.li.pdf", "wills_manual_deepseek.json")

