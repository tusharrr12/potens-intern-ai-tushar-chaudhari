from pathlib import Path
from typing import List, Dict

import fitz
from backend.services.vector_store import (
    store_chunks,
    reset_vector_store,
)
from backend.config import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.services.vector_store import store_chunks

def is_valid_page(text: str) -> bool:
    """
    Returns True if the extracted page contains
    meaningful content for retrieval.
    """

    if len(text.strip()) < 80:
        return False

    blocked_words = [
        "table of contents",
        "contents",
        "index",
        "this page intentionally left blank",
    ]

    text_lower = text.lower()

    for word in blocked_words:
        if word in text_lower:
            return False

    return True

def extract_pdf_pages(pdf_path: Path) -> List[Dict]:
    """
    Extract text from a PDF page by page.

    Returns:
        List of dictionaries containing page text and metadata.
    """

    pages = []

    # Open the PDF
    document = fitz.open(pdf_path)

    # Company name comes from parent folder
    company = pdf_path.parent.name

    # Model name comes from file name
    model = (
        pdf_path.stem
        .replace("_", " ")
        .replace("-", " ")
    )

    # Read every page
    for page_number, page in enumerate(document, start=1):

        text = page.get_text().strip()

        # Skip completely empty pages
        if not is_valid_page(text):
            continue

        pages.append(
            {
                "text": text,
                "metadata": {
                    "company": company,
                    "model": model,
                    "source_file": pdf_path.name,
                    "page": page_number,
                },
            }
        )

    document.close()

    return pages


def discover_pdfs() -> List[Path]:
    """
    Discover all PDF files inside data/manuals recursively.

    Returns:
        List of PDF file paths.
    """

    manuals_path = settings.DATA_DIR / "manuals"

    pdf_files = []

    for pdf in manuals_path.rglob("*.pdf"):
        pdf_files.append(pdf)

    return pdf_files

def load_documents() -> List[Dict]:
    """
    Load all PDFs and extract their pages.

    Returns:
        List of page dictionaries containing text and metadata.
    """

    documents = []

    pdf_files = discover_pdfs()

    print(f"[INFO] Found {len(pdf_files)} PDF(s).")

    for pdf in pdf_files:

        print(f"[INFO] Processing: {pdf.name}")

        pages = extract_pdf_pages(pdf)

        documents.extend(pages)

    print(f"[INFO] Extracted {len(documents)} pages.")

    return documents

def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """
    Split extracted pages into smaller chunks while
    preserving metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    chunks = []
    chunk_counter = 1

    for document in documents:

        text = document["text"]

        metadata = document["metadata"]

        page_chunks = splitter.split_text(text)

        for chunk in page_chunks:

            chunk_metadata = metadata.copy()

            chunk_metadata["chunk_id"] = f"chunk_{chunk_counter:04d}"

            chunks.append(
                {
                    "text": chunk,
                    "metadata": chunk_metadata,
                }
            )

            chunk_counter += 1

    return chunks

def main():
    """
    Complete ingestion pipeline.
    """

    print("=" * 60)
    print("Starting document ingestion...")
    print("=" * 60)

    # Step 1
    documents = load_documents()

    # Step 2
    chunks = chunk_documents(documents)

    print(f"[INFO] Documents Loaded : {len(documents)}")
    print(f"[INFO] Chunks Created   : {len(chunks)}")

    # Step 3
    reset_vector_store()

    # Step 4
    store_chunks(chunks)

    print("\nIngestion completed successfully.")


if __name__ == "__main__":
    main()