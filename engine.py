import glob
import os
import pathlib
import openai
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from structures.entities import Topics

class Text:
    def __init__(self, content) -> None:
        self.content = content
        self.embeddings = np.array([])

    def create_embeddings(self, model):
        if self.embeddings.shape == (0,):
            self.embeddings = model.encode(self.content)

    def similarity(self, folder): #todo, change for something more general
        return cosine_similarity(self.embeddings.reshape(1, -1), folder.embeddings)[0]
    
    def top_N_similar(self, folder, N=10):
        similarities = self.similarity(folder)
        sorted_pairs = sorted(zip(folder.documents, similarities), key=lambda x: x[1], reverse=True)
        return [document for document, similarity in sorted_pairs[:N]]

    def as_struct(self, struct: object):
        return struct().source(f"Fill the arguments using the following text (use the same language): {self.content}")

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
    
class TextDocument(Document):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        with open(path, "r", encoding="utf-8") as f:
            self.content =Text(f.read())

    def create_embeddings(self, model, *args):
        return self.content.create_embeddings(model)
    
def document_router(path: str):
        file_extension = pathlib.Path(path).suffix
        if file_extension == ".txt":
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
        
    def load_folder(self, name: str, path: str):
        self.create_folder(name)
        files = glob.glob(f"{path}/*")
        for file in files:
            self.folders[name].add_document(file)

class Engine:
    def __init__(self, model):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.library = Library()
        self.embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    def query(self, prompt: str, max_tokens: int=256, temperature: float = 0):
        answer = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )["choices"][0]["message"]["content"]
        return Text(answer)
    
    def query_folder(self, prompt: str, folder: str, max_tokens: int=256, temperature: float = 0):
        folder = self.library.folders[folder]
        prompt_content = prompt
        prompt = Text(prompt)
        prompt.create_embeddings(self.embeddings_model)
        folder.create_embeddings(self.embeddings_model)
        documents_string = "<Documents>\n"
        for document in prompt.top_N_similar(folder, N=5):
            prompt_content += document.content.content + "\n\n"
        documents_string += "\n</Documents>"
        print("{documents_string}\n---\nUsing the documents, answer the user query: {prompt_content}")
        return self.query(f"{documents_string}\n---\nUsing the documents, answer the user query: {prompt_content}")
