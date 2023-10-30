import pytesseract
from pdf2image import convert_from_path
import markdownify
from bs4 import BeautifulSoup
import requests


def pdf_to_text(pdf_path):
    # Convert the PDF to images
    images = convert_from_path(pdf_path)

    # Extract text from each image
    texts = []
    for image in images:
        text = pytesseract.image_to_string(image)
        texts.append(text)

    # Combine all the text into one string
    full_text = "\n".join(texts)

    return full_text.replace("\n\n", "\n")


def clean_html(url: str) -> str:
    soup = BeautifulSoup(requests.get(url).text, "lxml")
    head = soup.head.get_text(strip=True)
    body = soup.body.get_text(strip=True)
    return f"""URL: {url}\nHEADER: {head}\n\n---\n\nContent: {body}"""
