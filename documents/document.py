import pathlib
import numpy as np
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity


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
            zip(folder.chunks, similarities), key=lambda x: x[1], reverse=True
        )
        return [chunk for chunk, similarity in sorted_pairs[:N]]

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


class Chunk(Text):
    def __init__(self, content, document, i) -> None:
        super().__init__(content)
        self.document = document
        self.i = i
        self.name = f"{self.document.path.split('/')[-1]}-{i}"

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
        string_formated += (
            f"<{name}: {self.name}>\n{self.content}\n</{name}: {self.name}>\n\n"
        )
        return string_formated


class Document:
    def __init__(
        self, content, chunking_strategy: dict, path: str = "No path specified"
    ) -> None:
        self.content = self.chunks(content, chunking_strategy)
        self.embeddings = np.array([])
        self.path = path

    def create_embeddings(self, model, *args):
        list_embeddings = []
        for chunk in self.content:
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
        for chunk in self.content:
            string_formated += (
                f"<{name}: {chunk.name}>\n{chunk.content}\n</{name}: {chunk.name}>\n\n"
            )
        return string_formated

    def chunks(self, content, chunking_strategy: str):
        strategy = chunking_strategy["strategy"]
        if strategy == "tokens":
            max_tokens = chunking_strategy["max_tokens"]
            raise NotImplementedError
        if strategy == "pattern":
            pattern = chunking_strategy["pattern"]
            return [
                Chunk(chunk_content, self, i)
                for i, chunk_content in enumerate(content.split(pattern))
            ]

    def __str__(self) -> str:
        chunks_str = "\n".join([str(chunk) for chunk in self.content])
        return f"{self.path} ({len(self.content)} chunks)\n---\n{chunks_str}"

    def __repr__(self) -> str:
        return f"{self.__class__}(name={self.path})"


class TextDocument(Document):
    def __init__(self, path: str, chunking_strategy: dict) -> None:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        self.path = path
        super().__init__(content, chunking_strategy, path=path)


class TeamsTranscriptDocument(Document):
    def __init__(self, path: str, chunking_strategy: dict) -> None:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            self.transcript = self.parse_transcript(content)
            transcript_str = ""
            for turn in self.transcript:
                transcript_str += f"{turn['speaker']}: {turn['text']}\n\n"
        self.path = path
        super().__init__(transcript_str, chunking_strategy, path=path)

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


def document_router(path: str, chunking_strategy: dict):
    file_extension = pathlib.Path(path).suffix
    if file_extension == ".txt":
        return TextDocument(path, chunking_strategy[".txt"])
    if file_extension == ".py":
        return TextDocument(path, chunking_strategy[".py"])
    if file_extension == ".vtt":
        return TeamsTranscriptDocument(path, chunking_strategy[".vtt"])
    pass


class Prompt(Text):
    def __init__(self, content: str) -> None:
        super().__init__(content)
        self.original_query = content
        self.linked_documents = []

    def add_document(self, document: Document):
        print(document)
        print("@@@@@@@@@@@@@@@@@@@@")
        self.linked_documents.append(document)
        documents_part = "Listing documents: \n---\n"
        for document in self.linked_documents:
            documents_part += document.formated + "\n"
        documents_part += "\n---\nUsing the documents, answer the user query: "
        self.content = f"{documents_part}{self.original_query}"
