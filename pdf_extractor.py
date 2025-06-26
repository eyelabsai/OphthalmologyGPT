from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LAParams, LTAnno, LTTextLine, LTFigure, LTLink
import json

def extract_links_and_content(pdf_path):
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextBoxHorizontal, LTTextLineHorizontal, LAParams, LTLink

    results = []

    laparams = LAParams()
    for page_num, layout in enumerate(extract_pages(pdf_path, laparams=laparams)):
        for element in layout:
            if isinstance(element, LTTextBoxHorizontal):
                for line in element:
                    if isinstance(line, LTTextLineHorizontal):
                        text = line.get_text().strip()
                        # Look for link annotations
                        for child in line:
                            if isinstance(child, LTLink):
                                link = child.uri
                                if link and text:
                                    results.append({
                                        "page": page_num + 1,
                                        "link_text": text,
                                        "url": link
                                    })

    return results

# Save results
def save_json(data, out_file="pdf_links_output.json"):
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# Example usage
if __name__ == "__main__":
    pdf_file = "your_file.pdf"
    links = extract_links_and_content(pdf_file)
    save_json(links)
    print(f"✅ Extracted {len(links)} hyperlinks.")
