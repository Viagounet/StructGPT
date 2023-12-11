import glob
import importlib
import sys
from typing import List
import numpy as np
import tiktoken
import requests

from sklearn.metrics.pairwise import cosine_similarity
from documents.parsing_utils import (
    clean_html,
    get_outline_pdf,
    pdf_to_text,
    pptx_to_text,
    readable_pdf_to_text,
    thread_parsing,
)


class Text:
    def __init__(self, content) -> None:
        self.content = content
        self.embeddings = np.array([])

    def create_embeddings(self, model):
        if self.embeddings.shape == (0,):
            self.embeddings = model.encode(self.content)

    def similarity(self, documents):  # todo, change for something more general
        embeddings_array = []
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
        self.short_content = None
        self.i = i
        if "WebDocument" in str(self.document.__class__):
            self.name = self.document.path
        elif "ImageBoardThreadDocument" in str(self.document.__class__):
            self.name = f"4chan thread: {self.document.path}"
        else:
            self.name = f"{self.document.path.split('/')[-1]}-{i}"

    @property
    def short_formated(self):
        string_formated = ""
        name = (
            str(self.document.__class__)
            .split(".")[-1]
            .replace(">", "")
            .replace("<", "")
            .replace("'", "")
        )
        string_formated += (
            f"<{name}: {self.name}>\n{self.short_content}\n</{name}: {self.name}>\n\n"
        )
        return string_formated

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
    """This class is the generic class to handle documents. It provides methods to initialize a document, split it into small chunks and search inside it."""

    def __init__(self, content, path: str = "No path specified") -> None:
        """
        Initializes the Document object
        Args:
            content (str): The text content of the document
            path (str, optional): The path to the document linked. Defaults to "No path specified".
        """
        self.content = content
        self.embeddings = np.array([])
        self.path = path

        # Handling 'special' document types
        if "::" in self.path:
            self.document_type = self.path.split("::")[0]
        else:
            self.document_type = self.path.split(".")[-1]

    def create_embeddings(self, model, *args):
        """Creates text embeddings for each chunk. These embeddings will then be used to find similar chunks.

        Args:
            model (str): HuggingFace similarity model path
        """
        list_embeddings = []
        for chunk in self.chunks:
            chunk.create_embeddings(model)
            list_embeddings.append(chunk.embeddings)
        self.embeddings = np.array(list_embeddings)

    @property
    def formated(self):
        """
        Returns:
            str: The formated string of the document, with its name & chunks content.
        """
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

    def chunking(
        self, chunking_parameters: dict, post_processing_parameters: dict
    ) -> List[Chunk]:
        """This is a generic method to split the document into several chunks and process them.
        See process/chunking_strategy & process/post_processing python files

                Args:
                    chunking_parameters (dict): The chunking parameters from the parameters.yaml file
                    post_processing_parameters (dict): The post-processing parameters from the parameters.yaml file

                Returns:
                    List[Chunk]: The chunks of the document
        """
        chunking_strategy = chunking_parameters[self.document_type]["strategy"]
        chunking_kwargs = chunking_parameters[self.document_type]["kwargs"]
        post_processing_strategy = post_processing_parameters["strategy"]
        post_processing_kwargs = post_processing_parameters["kwargs"]
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

    def search_chunks(self, search_request: str) -> List[Chunk]:
        """Searches for all chunks that contain an expression

        Args:
            search_request (str): The expression to search for

        Returns:
            List[Chunk]: The list of chunks that contain the expression
        """
        return [
            chunk
            for chunk in self.chunks
            if search_request.lower() in chunk.content.lower()
        ]

    @property
    def n_tokens(self) -> int:
        """Returns the number of GPT tokens used for displaying the formated document.

        Returns:
            int: the number of GPT tokens used for displaying the formated document.
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(self.formated))

    @property
    def metadata(self):
        n_words = self.content.count(" ")
        n_letters = len(self.content)
        n_chunks = len(self.chunks)
        return {
            "path": self.path,
            "words": n_words,
            "letters": n_letters,
            "chunks": n_chunks,
            "document type": self.document_type,
        }

    def read_page(self, page: int) -> str:
        """Returns the content of the nth page. Here, does not return anything. See the PDFDocuments overloading implementation.

        Args:
            page (int): Page index

        Returns:
            str: "Not implemented"
        """
        return "Not implemented."

    def __str__(self) -> str:
        chunks_str = "\n".join([str(chunk) for chunk in self.chunks])
        return f"{self.path} ({len(self.chunks)} chunks)\n---\n{chunks_str}"

    def __repr__(self) -> str:
        return f"{self.__class__}(name={self.path})"


def document_router(path: str, convert_strategy: dict):
    """This function will route a file path to its corresponding Document class

    Args:
        path (str): The document path
        convert_strategy (dict): The convert strategy dictionnary from the parameters.yaml

    Returns:
        Document: The correct Document class
    """
    if "::" in path:
        document_type = path.split("::")[0]
    else:
        document_type = path.split(".")[-1]
    MyObjectClass = getattr(
        sys.modules["documents.definitions.definitions"],
        convert_strategy[document_type]["type"],
    )
    return MyObjectClass(path)


class Folder:
    """The Folder class, where we store Document objects."""

    def __init__(self, parameters: dict):
        """We initialize the folder with the relevant chunking parameters and initialize with null embeddings.

        Args:
            parameters (dict): parameters.yaml parameters
        """
        self.documents = []
        self.embeddings = np.array([])
        self.convert_strategy = parameters["processing"]["convert"]
        self.chunking_strategy = parameters["processing"]["chunking_strategy"]
        self.chunks_post_processing_strategy = parameters["processing"][
            "chunks_post_processing"
        ]

        self.chunks = []

    def add_document(self, path: str, get_chunks=True):
        """Adds a Document

        Args:
            path (str): The document path
            get_chunks (bool, optional): If it should split the document into chunks, might take a while for long documents. Defaults to True.
        """
        document = document_router(path, self.convert_strategy)
        if "::" in path:
            document_type = path.split("::")[0]
        else:
            document_type = path.split(".")[-1]
        post_processing_strategy = self.chunks_post_processing_strategy[document_type]
        if get_chunks:
            document.chunks = document.chunking(
                self.chunking_strategy, post_processing_strategy
            )
        else:
            document.chunks = []
        if document:
            self.documents.append(document)
            self.chunks += document.chunks

    def create_embeddings(self, model, *args):
        """Generates embeddings for every Document.

        Args:
            model (str): HuggingFace similarity model path
        """
        embeddings_array = []
        for document in self.documents:
            document.create_embeddings(model)
            embeddings_array += document.embeddings.tolist()
        self.embeddings = np.array(embeddings_array)

    def __repr__(self):
        return f"Folder(documents={self.documents})"


class Prompt(Text):
    """The Prompt class, used to manage the what will be sent to GPT. Provides methods to add chunks or full documents to the prompt.
    Args:
        Text (_type_): _description_
    """

    def __init__(self, content: str) -> None:
        super().__init__(content)
        self.original_query = content
        self.linked_documents = []

    def add_document(self, document: Document):
        """Adds a document to the prompt in a nicely formated manner.

        Args:
            document (Document): The Document object to add.
        """
        self.linked_documents.append(document)
        documents_part = "Listing documents: \n---\n"
        for document in self.linked_documents:
            documents_part += document.formated
        documents_part += "\n---\nUsing the documents, answer the user query in the same language as his: "
        self.content = f"{documents_part}{self.original_query}"

    def add_chunk(self, chunk: Chunk, short=False):
        """Adds a document chunk to the prompt in a nicely formated manner.


        Args:
            chunk (Chunk): A Document Chunk.
            short (bool, optional): If we should use the summarized version of the Chunk. Defaults to False.
        """
        self.linked_documents.append(chunk)
        documents_part = "Listing documents: \n---\n"
        for chunk in self.linked_documents:
            if short:
                documents_part += chunk.short_formated
            else:
                documents_part += chunk.formated
        documents_part += "\n---\nUsing the documents, answer the user query in the same language as his: "
        self.content = f"{documents_part}{self.original_query}"

    def reset(self):
        """Deletes all documents from the prompt."""
        self.linked_documents = []

    @property
    def tokens(self):
        """The total number of GPT tokens used by the prompt.

        Returns:
            int: The total number of GPT tokens used by the prompt.
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(self.content))


class Library:
    """
    The library stores and manages files.
    """

    def __init__(self, parameters):
        self.folders = {}
        self.parameters = parameters

    def create_folder(self, name: str):
        """Creates a folder in the library.

        Args:
            name (str): The folder name.

        Raises:
            KeyError: Raises an error if the name already exists.
        """
        if name not in self.folders:
            self.folders[name] = Folder(self.parameters)
        else:
            raise KeyError

    def load_folder(self, name: str, path: str, get_chunks=True):
        self.create_folder(name)
        files = glob.glob(f"{path}/*")
        for file in files:
            self.folders[name].add_document(file, get_chunks)

    # This method is to move to another class
    def web_search(
        self,
        query: str,
        api_key: str,
        cse_id: str,
        n_results=10,
        skip_files=False,
        **kwargs,
    ):
        """Performs a web search useing the Google Search API

        Args:
            query (str): The query you would like to search on Google
            api_key (str): Google Search API key
            cse_id (str): Google Search API CSE id
            n_results (int, optional): The number of websites the method returns. Defaults to 10.
            skip_files (bool, optional): Skip urls that are files. Defaults to False.

        Returns:
            List[str]: Top nth urls from the search
        """
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
