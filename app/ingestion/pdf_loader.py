"""
Phase 1: PDF Ingestion.

Extracts text from a PDF, page by page, and tags every page with
metadata (filename, page_number). This metadata is the seed for every
citation shown later in the pipeline -- get it right here and citations
downstream are basically free.
"""
import os
import fitz  # PyMuPDF


def extract_pages(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF, one entry per page.

    Returns:
        List of dicts: [{"text": str, "filename": str, "page_number": int}, ...]
        page_number is 1-indexed (human-friendly, matches what a user sees
        in a PDF viewer).
    """
    filename = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)

    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        text = _clean_text(text)
        if not text.strip():
            continue  # skip blank pages (e.g. cover separators, images-only)
        pages.append({
            "text": text,
            "filename": filename,
            "page_number": i + 1,
        })

    doc.close()
    return pages


def _clean_text(text: str) -> str:
    """
    Light cleanup: collapse excessive whitespace/newlines that PDF
    extraction tends to introduce, without destroying paragraph structure.
    """
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]  # drop empty lines
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_loader.py <path_to_pdf>")
        sys.exit(1)

    pages = extract_pages(sys.argv[1])
    print(f"Extracted {len(pages)} pages from {sys.argv[1]}")
    for p in pages[:2]:
        print(f"\n--- Page {p['page_number']} ---")
        print(p["text"][:300])
