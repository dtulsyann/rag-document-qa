"""
Phase 2 (baseline): Fixed-size token chunking with overlap.

Splits each page's text into chunks of ~CHUNK_SIZE tokens, with
CHUNK_OVERLAP tokens shared between consecutive chunks so context isn't
cut off mid-sentence at boundaries.

Note: a single chunk can span content that "belongs" to one page only,
since we chunk per-page (not across page boundaries). This keeps page
citations exact -- a deliberate tradeoff over chunking the whole document
as one long stream, where a chunk could straddle two pages and force a
page_start/page_end citation instead.
"""
from app.chunking.tokenizer import encode, decode
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_pages(pages: list[dict], chunk_size: int = CHUNK_SIZE,
                 overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Args:
        pages: output of pdf_loader.extract_pages()
        chunk_size: target tokens per chunk
        overlap: tokens shared between consecutive chunks

    Returns:
        List of chunk dicts:
        [{"chunk_id", "text", "filename", "page_number", "chunk_index"}, ...]
    """
    all_chunks = []

    for page in pages:
        tokens = encode(page["text"])
        chunk_index = 0
        start = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = decode(chunk_tokens)

            chunk_id = f"{page['filename']}_p{page['page_number']}_c{chunk_index}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "filename": page["filename"],
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
            })

            chunk_index += 1
            if end == len(tokens):
                break
            start = end - overlap  # step forward, keeping `overlap` tokens shared

    return all_chunks


if __name__ == "__main__":
    from app.ingestion.pdf_loader import extract_pages
    import sys

    pages = extract_pages(sys.argv[1])
    chunks = chunk_pages(pages)
    print(f"Produced {len(chunks)} chunks from {len(pages)} pages")
    print(chunks[0])
