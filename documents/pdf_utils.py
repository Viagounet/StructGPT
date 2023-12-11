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


def format_sections(sections, level=0):
    formatted_text = ""
    indent = "  " * level  # Two spaces per level of indentation

    for section in sections:
        if "number" in section and "name" in section and "page" in section:
            formatted_text += f"{indent}Section {section['number']}: {section['name']} (Page {section['page']})\n"
        if "subsections" in section:
            formatted_text += format_sections(section["subsections"], level + 1)

    return formatted_text
