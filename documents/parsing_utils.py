import pytesseract
from pdf2image import convert_from_path
from bs4 import BeautifulSoup
import requests
import urllib.request
from pptx import Presentation


def pdf_to_text(pdf_path: str) -> str:
    """Performs the OCR of the pdf using Tesseract

    Args:
        pdf_path (str): The path of the pdf

    Returns:
        str: The raw text from the pdf
    """
    # Convert the PDF to images
    images = convert_from_path(pdf_path, dpi=72, size=(840, 1190))

    # Extract text from each image
    texts = []
    for image in images:
        text = pytesseract.image_to_string(image)
        if len(text) > 2:
            texts.append(text)

    # Combine all the text into one string
    full_text = "\n".join(texts)

    return full_text.replace("\n\n", "\n")


def readable_pdf_to_text(doc):
    """Reads a fitz document.

    Args:
        doc (_type_): _description_

    Returns:
        str: The raw text from the pdf
    """
    full_text = ""
    for page in doc:  # iterate the document pages
        full_text += page.get_text()
    return full_text.replace("\n\n", "\n")


def get_outline_pdf(pdf_path):
    """
    To do
    """
    return None


def clean_html(url: str) -> str:
    """
    Gets the clean html content of a webpage
    """
    head = None
    body = None
    try:
        fp = urllib.request.urlopen(url)
        content = fp.read()
        soup = BeautifulSoup(content, "lxml")
        head = soup.head.get_text(strip=True)
        body = soup.body.get_text(strip=True)
        return f"""URL: {url}\nHEADER: {head}\n\n---\n\nContent: {body}"""
    except Exception as ex:
        return f"""URL: {url}\nHEADER: {head}\n\n---\n\nContent: {body} - {ex}"""


def thread_parsing(thread_id: str) -> str:
    """Parses an imageboard thread into readable text

    Args:
        thread_id (str): The imageboard thread id

    Returns:
        str: Raw text of the thread
    """
    chan, board, thread = thread_id.split(":")
    url = f"https://boards.4channel.org/{board}/thread/{thread}"
    soup = BeautifulSoup(requests.get(url).text, "lxml")

    post_message_tags = soup.find_all(class_="postMessage")

    for tag in post_message_tags:
        br_tags = tag.find_all("br")
        for br in br_tags:
            br.replace_with("\n")
    thread_string = ""
    # Extract and print the modified text inside each "postMessage" tag
    for tag in post_message_tags:
        thread_string += tag.get_text() + "\n"
        thread_string += "------------\n"
    return thread_string


def pptx_to_text(path):
    txt = ""
    prs = Presentation(path)

    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                print(shape.left)
                txt += shape.text + "\n"
        txt += "\n----------\n"
    return txt
