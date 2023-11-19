import glob
import importlib
import pathlib
import sys
import numpy as np
import tiktoken
import requests

from sklearn.metrics.pairwise import cosine_similarity
from documents.parsing_utils import clean_html, pdf_to_text, thread_parsing
from process.chunking_strategy import *


class Text:
    def __init__(self, content) -> None:
        self.content = content
        self.embeddings = np.array([])

    def create_embeddings(self, model):
        if self.embeddings.shape == (0,):
            self.embeddings = model.encode(self.content)

    def similarity(self, documents):  # todo, change for something more general
        embeddings_array = []
        print("--------------------")
        for document in documents:
            embeddings_array += document.embeddings.tolist()
        embeddings_array = np.array(embeddings_array)
        if len(embeddings_array.shape) == 1:
            embeddings_array = embeddings_array.reshape(
                int(embeddings_array.shape[0] / 384), 384
            )
        return cosine_similarity(self.embeddings.reshape(1, -1), embeddings_array)[0]

    def top_N_similar(self, documents, N=10):
        chunks = []
        for document in documents:
            chunks += document.chunks
        similarities = self.similarity(documents)
        sorted_pairs = sorted(
            zip(chunks, similarities), key=lambda x: x[1], reverse=True
        )
        return [chunk for chunk, similarity in sorted_pairs[:N]]

    def as_struct(self, struct: object, model):
        return struct().source(
            f"Fill the arguments using the following text (use the same language): {self.content}",
            model,
            model_type="engine",
        )

    @property
    def formated(self):
        return f"TextDocument:\nContent: {self.content}"

    @property
    def n_tokens(self):
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(self.content))

    def __str__(self) -> str:
        return self.content

    def __repr__(self) -> str:
        return self.content


class Chunk(Text):
    def __init__(self, content, document, i) -> None:
        super().__init__(content)
        self.document = document
        self.i = i
        if "WebDocument" in str(self.document.__class__):
            self.name = self.document.path
        elif "ImageBoardThreadDocument" in str(self.document.__class__):
            self.name = f"4chan thread: {self.document.path}"
        else:
            self.name = f"{self.document.path.split('/')[-1]}-{i}"

    @property
    def formated(self):
        string_formated = ""
        name = (
            str(self.document.__class__)
            .split(".")[-1]
            .replace(">", "")
            .replace("<", "")
            .replace("'", "")
        )
        string_formated += (
            f"<{name}: {self.name}>\n{self.content}\n</{name}: {self.name}>\n\n"
        )
        return string_formated


class Document:
    def __init__(self, content, path: str = "No path specified") -> None:
        self.content = content
        self.embeddings = np.array([])
        self.path = path
        if "::" in self.path:
            self.document_type = self.path.split("::")[0]
        else:
            self.document_type = self.path.split(".")[-1]

    def create_embeddings(self, model, *args):
        list_embeddings = []
        for chunk in self.chunks:
            chunk.create_embeddings(model)
            list_embeddings.append(chunk.embeddings)
        self.embeddings = np.array(list_embeddings)

    @property
    def formated(self):
        string_formated = ""
        name = (
            str(self.__class__)
            .split(".")[-1]
            .replace(">", "")
            .replace("<", "")
            .replace("'", "")
        )
        for chunk in self.chunks:
            string_formated += (
                f"<{name}: {chunk.name}>\n{chunk.content}\n</{name}: {chunk.name}>\n\n"
            )
        return string_formated

    def chunking(self, chunking_parameters: dict, post_processing_parameters: dict):
        chunking_strategy = chunking_parameters[self.document_type]["strategy"]
        chunking_kwargs = chunking_parameters[self.document_type]["kwargs"]
        print(post_processing_parameters)
        post_processing_strategy = post_processing_parameters["strategy"]
        post_processing_kwargs = post_processing_parameters["kwargs"]
        print(post_processing_kwargs)
        chunking_module = importlib.import_module(
            f"process.chunking_strategy.{chunking_strategy}"
        )
        chunking_function = chunking_module.func
        chunks = chunking_function(self, **chunking_kwargs)

        post_processing_module = importlib.import_module(
            f"process.post_processing.{post_processing_strategy}"
        )
        post_processing_function = post_processing_module.post_processing

        chunks = chunking_function(self, **chunking_kwargs)

        new_chunks = []
        for chunk in chunks:
            chunk.content = post_processing_function(
                chunk.content, **post_processing_kwargs
            )
            new_chunks.append(chunk)
        return new_chunks

    def __str__(self) -> str:
        chunks_str = "\n".join([str(chunk) for chunk in self.chunks])
        return f"{self.path} ({len(self.content)} chunks)\n---\n{chunks_str}"

    def __repr__(self) -> str:
        return f"{self.__class__}(name={self.path})"


class TextDocument(Document):
    def __init__(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        self.path = path
        super().__init__(content, path=path)


class PDFDocument(Document):
    def __init__(self, path) -> None:
        content = pdf_to_text(path)
        self.path = path
        super().__init__(content, path)


class WebDocument(Document):
    def __init__(self, url: str) -> None:
        self.path = url.split("::")[1]
        print(self.path)
        content = clean_html(self.path)
        super().__init__(content, url)


class ImageBoardThreadDocument(Document):
    def __init__(self, thread_id: str) -> None:
        self.path = id
        content = thread_parsing(thread_id)
        super().__init__(content, thread_id)


class TeamsTranscriptDocument(Document):
    def __init__(
        self,
        path: str,
    ) -> None:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            self.transcript = self.parse_transcript(content)
            transcript_str = ""
            for turn in self.transcript:
                transcript_str += f"{turn['speaker']}: {turn['text']}\n\n"
        self.path = path
        super().__init__(transcript_str, path=path)

    def parse_transcript(self, transcript):
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


def document_router(path: str, convert_strategy: dict):
    if "::" in path:
        document_type = path.split("::")[0]
    else:
        document_type = path.split(".")[-1]
    MyObjectClass = getattr(
        sys.modules[__name__], convert_strategy[document_type]["type"]
    )
    return MyObjectClass(path)


class Folder:
    def __init__(self, parameters: dict):
        self.documents = []
        self.embeddings = np.array([])
        self.convert_strategy = parameters["processing"]["convert"]
        self.chunking_strategy = parameters["processing"]["chunking_strategy"]
        self.chunks_post_processing_strategy = parameters["processing"][
            "chunks_post_processing"
        ]

        self.chunks = []

    def add_document(self, path: str):
        document = document_router(path, self.convert_strategy)
        if "::" in path:
            document_type = path.split("::")[0]
        else:
            document_type = path.split(".")[-1]
        post_processing_strategy = self.chunks_post_processing_strategy[document_type]
        document.chunks = document.chunking(
            self.chunking_strategy, post_processing_strategy
        )
        if document:
            self.documents.append(document)
            self.chunks += document.chunks

    def create_embeddings(self, model, *args):
        embeddings_array = []
        for document in self.documents:
            document.create_embeddings(model)
            embeddings_array += document.embeddings.tolist()
        self.embeddings = np.array(embeddings_array)

    def __repr__(self):
        return f"Folder(documents={self.documents})"


class Prompt(Text):
    def __init__(self, content: str) -> None:
        super().__init__(content)
        self.original_query = content
        self.linked_documents = []

    def add_document(self, document: Document):
        self.linked_documents.append(document)
        documents_part = "Listing documents: \n---\n"
        for document in self.linked_documents:
            documents_part += document.formated
        documents_part += "\n---\nUsing the documents, answer the user query in the same language as his: "
        self.content = f"{documents_part}{self.original_query}"

    def add_chunk(self, chunk: Chunk):
        self.linked_documents.append(chunk)
        documents_part = "Listing documents: \n---\n"
        for chunk in self.linked_documents:
            documents_part += chunk.formated
        documents_part += "\n---\nUsing the documents, answer the user query in the same language as his: "
        self.content = f"{documents_part}{self.original_query}"


class Library:
    """
    The library stores and manages files.
    """

    def __init__(self, parameters):
        self.folders = {}
        self.parameters = parameters

    def create_folder(self, name: str):
        if name not in self.folders:
            self.folders[name] = Folder(self.parameters)
        else:
            raise KeyError

    def load_folder(self, name: str, path: str):
        self.create_folder(name)
        files = glob.glob(f"{path}/*")
        for file in files:
            self.folders[name].add_document(file)

    def web_search(
        self, query, api_key, cse_id, n_results=10, skip_files=False, **kwargs
    ):
        search_url = "https://www.googleapis.com/customsearch/v1"
        params = {"q": query, "key": api_key, "cx": cse_id, "num": n_results}
        params.update(kwargs)
        response = requests.get(search_url, params=params)
        result = response.json()
        websites = [l["link"] for l in result["items"]] if "items" in result else []
        websites_filtered = [
            w for w in websites if w.split(".")[-1] not in ["txt", "pdf", "png"]
        ]
        if skip_files:
            return websites_filtered
        return websites

    def __str__(self) -> str:
        return str(self.folders.values())
