from PyPDF2 import PdfReader

def read_pdf(filename):
    reader = PdfReader(filename)
    #print(f"PDF file is loaded, amount of pages: {len(reader.pages)}")
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text