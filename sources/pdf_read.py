from PyPDF2 import PdfReader
from typing import BinaryIO

def read_pdf(filename: BinaryIO) -> str:
    """
    Reads a PDF from a file-like object and returns all text from all pages.

    Args:
    file (BinaryIO): A PDF file object (e.g. file.file from UploadFile).

    Returns:
    str: Text concatenated from all pages of the PDF. Empty pages are ignored.

    """
    reader = PdfReader(filename)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text