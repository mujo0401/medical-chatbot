import logging
from typing import List
from pathlib import Path

try:
    import pdfplumber
    import PyPDF2
    PDF_PROCESSING_AVAILABLE = True
except ImportError:
    PDF_PROCESSING_AVAILABLE = False

logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_path: str) -> str:
    """Try pdfplumber first; fallback to PyPDF2. Raise if neither works."""
    if not PDF_PROCESSING_AVAILABLE:
        raise RuntimeError("PDF libraries missing; install PyPDF2 and pdfplumber")

    text_content = ""
    path = Path(file_path)
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_content += f"\n--- Page {i+1} ---\n" + page_text + "\n"
    except Exception as e1:
        logger.warning(f"pdfplumber failed: {e1}, trying PyPDF2.")
        try:
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\n--- Page {i+1} ---\n" + page_text + "\n"
        except Exception as e2:
            logger.error(f"Both PDF methods failed: {e2}")
            raise RuntimeError(f"Failed to extract text from {file_path}: {e2}")

    if not text_content.strip():
        raise RuntimeError(f"No text extracted from {file_path}")
    return text_content

def read_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
