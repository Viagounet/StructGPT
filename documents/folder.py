import numpy as np
from documents.document import document_router


class Folder:
    def __init__(self, chunking_strategy: dict):
        self.documents = []
        self.embeddings = np.array([])
        self.chunking_strategy = chunking_strategy
        self.chunks = []

    def add_document(self, path: str):
        document = document_router(path, self.chunking_strategy)
        if document:
            self.documents.append(document)
            self.chunks += document.content

    def create_embeddings(self, model, *args):
        embeddings_array = []
        for document in self.documents:
            document.create_embeddings(model)
            embeddings_array += document.embeddings.tolist()
        self.embeddings = np.array(embeddings_array)

    def add_website(self, url: str):
        document = document_router(url, self.chunking_strategy)
        print(document)
        if document:
            self.documents.append(document)
            self.chunks += document.content

    def __repr__(self):
        return f"Folder(documents={self.documents})"
