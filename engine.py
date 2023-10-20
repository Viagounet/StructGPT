from datetime import datetime
import glob
import os
import pathlib
import openai
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

from structures.entities import Topics


class Text:
    def __init__(self, content) -> None:
        self.content = content
        self.embeddings = np.array([])

    def create_embeddings(self, model):
        if self.embeddings.shape == (0,):
            self.embeddings = model.encode(self.content)

    def similarity(self, folder):  # todo, change for something more general
        return cosine_similarity(self.embeddings.reshape(1, -1), folder.embeddings)[0]

    def top_N_similar(self, folder, N=10):
        similarities = self.similarity(folder)
        sorted_pairs = sorted(
            zip(folder.documents, similarities), key=lambda x: x[1], reverse=True
        )
        return [document for document, similarity in sorted_pairs[:N]]

    def as_struct(self, struct: object):
        return struct().source(
            f"Fill the arguments using the following text (use the same language): {self.content}"
        )

    @property
    def n_tokens(self):
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(self.content))

    def __str__(self) -> str:
        return self.content

    def __repr__(self) -> str:
        return self.content


class Document:
    def __init__(self, path: str) -> None:
        self.path = path
        self.content = Text("")
        self.embeddings = np.array([])

    def create_embeddings(self, *args):
        return None

    @property
    def formated(self):
        name = (
            str(self.__class__)
            .split(".")[-1]
            .replace(">", "")
            .replace("<", "")
            .replace("'", "")
        )
        return (
            f"<{name}: {self.path.split('/')[-1]}>\n{self.content.content}\n<{name}/>"
        )

    def __str__(self) -> str:
        return self.content

    def __repr__(self) -> str:
        return f"{self.__class__}(name={self.path})"


class ChunkDocument(Document):
    def __init__(self, name: str, content: str, i: int) -> None:
        super().__init__("None")
        self.content = Text(content)
        self.name = f"{name} - chunk {i}"

    def __repr__(self) -> str:
        return f"{self.__class__}(name={self.name})"

    def create_embeddings(self, model, *args):
        return self.content.create_embeddings(model)


class TextDocument(Document):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        with open(path, "r", encoding="utf-8") as f:
            self.content = Text(f.read())

    def create_embeddings(self, model, *args):
        return self.content.create_embeddings(model)


def document_router(path: str):
    file_extension = pathlib.Path(path).suffix
    if file_extension in [".txt", ".py"]:
        return TextDocument(path)
    pass


class Folder:
    def __init__(self):
        self.documents = []
        self.embeddings = np.array([])

    def add_document(self, path: str):
        document = document_router(path)
        if document:
            self.documents.append(document)

    def create_embeddings(self, model, *args):
        for document in self.documents:
            document.create_embeddings(model)
        self.embeddings = np.array([doc.content.embeddings for doc in self.documents])

    def __repr__(self):
        return f"Folder(documents={self.documents})"


class Library:
    """
    The library stores and manages files.
    """

    def __init__(self):
        self.folders = {}

    def create_folder(self, name: str):
        if name not in self.folders:
            self.folders[name] = Folder()
        else:
            raise KeyError

    def load_folder_from_chunks(self, name: str, chunks):
        self.create_folder(name)
        self.folders[name].documents = [
            ChunkDocument(name, chunk, i + 1) for i, chunk in enumerate(chunks)
        ]

    def load_folder(self, name: str, path: str):
        self.create_folder(name)
        files = glob.glob(f"{path}/*")
        for file in files:
            self.folders[name].add_document(file)

    def __str__(self) -> str:
        return str(self.folders.values())


class Prompt(Text):
    def __init__(self, content: str) -> None:
        super().__init__(content)
        self.original_query = content
        self.linked_documents = []

    def add_document(self, document: Document):
        self.linked_documents.append(document)
        documents_part = "Listing documents: \n---\n"
        for document in self.linked_documents:
            documents_part += document.formated + "\n"
        documents_part += "\n---\nUsing the documents, answer the user query: "
        self.content = f"{documents_part}{self.original_query}"


class AnswerLog:
    def __init__(self, prompt, answer, model, prices_prompt, prices_completion):
        now = datetime.now()
        self.date = now.strftime("%d/%m/%Y %H:%M:%S")
        self.prompt = prompt
        self.answer = answer
        self.prices = {
            "prompt": prompt.n_tokens * prices_prompt[model],
            "completion": answer.n_tokens * prices_completion[model],
            "total": (prompt.n_tokens * prices_prompt[model])
            + (answer.n_tokens * prices_completion[model]),
        }

    def __str__(self) -> str:
        return self.answer.content


class Engine:
    def __init__(self, model):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.library = Library()
        self.embeddings_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.logs = []
        self.prices_prompt = {
            "dai-semafor-nlp-gpt-35-turbo-model-fr": 0.002 / 1000,
            "dai-semafor-nlp-gpt-4-model-fr": 0.03 / 1000,
            "dai-semafor-nlp-gpt-4-32k-model-fr": 0.06 / 1000,
        }
        self.prices_completion = {
            "dai-semafor-nlp-gpt-35-turbo-model-fr": 0.002 / 1000,
            "dai-semafor-nlp-gpt-4-model-fr": 0.06 / 1000,
            "dai-semafor-nlp-gpt-4-32k-model-fr": 0.12 / 1000,
        }

    def query(self, prompt: str, max_tokens: int = 256, temperature: float = 0):
        prompt = Prompt(prompt)
        answer = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt.content}],
            max_tokens=max_tokens,
            temperature=temperature,
        )["choices"][0]["message"]["content"]
        answer = Text(answer)
        answer_log = AnswerLog(
            prompt, answer, self.model, self.prices_prompt, self.prices_completion
        )
        self.logs.append(answer_log)
        return answer_log

    def query_folder(
        self,
        prompt: str,
        folder: str,
        max_tokens: int = 256,
        temperature: float = 0,
        top_N=None,
    ):
        folder = self.library.folders[folder]
        prompt_content = prompt
        prompt = Prompt(prompt)
        prompt.create_embeddings(self.embeddings_model)
        folder.create_embeddings(self.embeddings_model)
        documents = folder.documents
        if top_N:
            documents = prompt.top_N_similar(folder, N=top_N)
        for document in documents:
            prompt.add_document(document)
        return self.query(prompt.content, max_tokens=max_tokens)

    def print_logs_history(self):
        logs_string = ""
        for log in self.logs:
            logs_string += f"- {log.date}\n\t- Prompt: {log.prompt.content}\n\t- \n\t- Answer: {log.answer.content}\n\t- Price: {round(log.prices['total'], 5)}$ (prompt -> {round(log.prices['prompt'], 5)} / completion -> {round(log.prices['completion'], 5)})\n\n---\n"
        print(logs_string)
