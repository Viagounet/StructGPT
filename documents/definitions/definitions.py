from typing import List
import tiktoken
import fitz

from documents.document import Chunk, Document
from documents.parsing_utils import (
    clean_html,
    get_outline_pdf,
    pdf_to_text,
    pptx_to_text,
    readable_pdf_to_text,
    thread_parsing,
)


class AnswerDocument(Document):
    """A GPT answer interpreted as a document

    Args:
        Document (Document): The Document class
    """

    def __init__(self, content: str, path="Model answer") -> None:
        self.path = path
        super().__init__(content, path=path)
        self.chunks = self.chunking()

    def chunking(self):
        """A special chunking strategy for this document, with a big chunk size. This method is overloading because we do not want the user to change it in the parameters.

        Returns:
            List[Chunk]: The list of chunks that contain the expression
        """
        pattern = "---\n"
        max_tokens = 2048
        string = self.content
        if pattern == "":
            turns = list(string)
        else:
            turns = string.split(pattern)
        encoding = tiktoken.get_encoding("cl100k_base")
        transcript_parts = []
        current_segment = ""

        for turn in turns:
            if len(encoding.encode(current_segment)) > max_tokens:
                transcript_parts.append(current_segment)
                current_segment = turn
            else:
                current_segment += turn + pattern
        transcript_parts.append(current_segment)
        return [
            Chunk(chunk_content, self, i)
            for i, chunk_content in enumerate(transcript_parts)
        ]


class TextDocument(Document):
    """For .txt documents

    Args:
        Document (_type_): _description_
    """

    def __init__(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        self.path = path
        super().__init__(content, path=path)


class PowerPointDocument(Document):
    """For .pptx documents

    Args:
        Document (_type_): _description_
    """

    def __init__(self, path: str = "No path specified") -> None:
        content = pptx_to_text(path)
        self.path = path
        super().__init__(content, path)


class PDFDocument(Document):
    """For any generic .pdf document. Will OCR every page using tesseract.

    Args:
        Document (_type_): _description_
    """

    def __init__(self, path) -> None:
        content = pdf_to_text(path)
        self.path = path
        super().__init__(content, path)


class PDFReadableDocument(Document):
    """For readable .pdf documents. Will use fitz to read the text content of the document.

    Args:
        Document (_type_): _description_
    """

    def __init__(self, path) -> None:
        self.fitz_document = fitz.open(path)  # open a document
        content = readable_pdf_to_text(self.fitz_document)
        self.path = path
        super().__init__(content, path)

    def read_page(self, page: int) -> str:
        """Reads the nth page of the document

        Args:
            page (int): page index

        Returns:
            str: The nth page content
        """
        return self.fitz_document[page + 1].get_text().replace("\n\n", "\n")


class WebDocument(Document):
    """For web pages documents

    Args:
        Document (_type_): _description_
    """

    def __init__(self, url: str) -> None:
        self.path = url.split("::")[1]
        content = clean_html(self.path)
        super().__init__(content, url)


class ImageBoardThreadDocument(Document):
    """For imageboards-inspired websites

    Args:
        Document (_type_): _description_
    """

    def __init__(self, thread_id: str) -> None:
        self.path = id
        content = thread_parsing(thread_id)
        super().__init__(content, thread_id)


class TeamsTranscriptDocument(Document):
    """For Teams transcript documents.

    Args:
        Document (_type_): _description_
    """

    def __init__(
        self,
        path: str,
    ) -> None:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            self.transcript = self.parse_transcript(
                content
            )  # We parse the document in a nice way for GPT
            transcript_str = ""
            for turn in self.transcript:
                transcript_str += f"{turn['speaker']}: {turn['text']}\n\n"
        self.path = path
        super().__init__(transcript_str, path=path)

    def parse_transcript(self, transcript: str) -> List[dict]:
        """_summary_

        Args:
            transcript (str): The raw transcript of the meeting

        Returns:
            List[dict]: A list of every speech turn, containing the speaker and associated text
        """
        result = []
        current_speaker = None
        current_text = ""

        for line in transcript.split("\n"):
            if line.startswith("<v"):
                speaker_start = line.find("<v") + 2
                speaker_end = line.find(">")
                speaker = line[speaker_start:speaker_end].strip()

                text_start = line.find(">", speaker_end) + 1
                text_end = line.find("</v>")
                text = line[text_start:text_end].strip()

                if current_speaker is None:
                    current_speaker = speaker
                    current_text = text
                elif current_speaker == speaker:
                    current_text += " " + text
                else:
                    result.append({"speaker": current_speaker, "text": current_text})
                    current_speaker = speaker
                    current_text = text

        if current_speaker is not None:
            result.append({"speaker": current_speaker, "text": current_text})

        return result
