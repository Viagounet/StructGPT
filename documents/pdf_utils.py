import pytesseract
from pdf2image import convert_from_path


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
