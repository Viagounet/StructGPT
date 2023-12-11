import re
from agent_functions.generic import AgentFunction


class DocumentMetaDataAF(AgentFunction):
    """Returns the Document metadata

    Args:
        AgentFunction (_type_): _description_
    """

    def __init__(self, name, description, engine) -> None:
        self.engine = engine
        super().__init__(name, description)

    def func(self, document_path: str) -> str:
        folder = document_path.split("/")[0]
        path = "/".join(document_path.split("/")[1:])
        target_document = None
        for document in self.engine.library.folders[folder].documents:
            if document.path == path:
                target_document = document
                break
        if not target_document:
            return f"{document_path} was not found in the library."
        metadata = []
        for key, value in document.metadata.items():
            metadata.append(f"{key}: {value}")

        return "\n".join(metadata)


class ReadDocumentAF(AgentFunction):
    """Reads the full Document content

    Args:
        AgentFunction (_type_): _description_
    """

    def __init__(self, name, description, engine) -> None:
        self.engine = engine
        super().__init__(name, description)

    def func(self, document_path: str) -> str:
        folder = document_path.split("/")[0]
        path = "/".join(document_path.split("/")[1:])
        target_document = None
        for document in self.engine.library.folders[folder].documents:
            if document.path == path:
                target_document = document
                break
        if not target_document:
            return f"{document_path} was not found in the library."
        if len(document.chunks) > 1:
            return f"{document_path} is too long and cannot be displayed at once. It is made of {len(document.chunks)} chunks. Please use the read_chunk() function if available. If not, end the program."
        return document.content


class ReadChunkAF(AgentFunction):
    """Reads a specific Document Chunk

    Args:
        AgentFunction (_type_): _description_
    """

    def __init__(self, name, description, engine) -> None:
        self.engine = engine
        super().__init__(name, description)

    def func(self, document_path: str, chunk_number: int) -> str:
        folder = document_path.split("/")[0]
        path = "/".join(document_path.split("/")[1:])
        target_document = None
        for document in self.engine.library.folders[folder].documents:
            if document.path == path:
                target_document = document
                break
        if not target_document:
            return f"{document_path} was not found in the library."
        return document.chunks[chunk_number].content


class ReadPageAF(AgentFunction):
    """Reads a PDFDocument page

    Args:
        AgentFunction (_type_): _description_
    """

    def __init__(self, name, description, engine) -> None:
        self.engine = engine
        super().__init__(name, description)

    def func(self, document_path: str, page_number: int) -> str:
        folder = document_path.split("/")[0]
        path = "/".join(document_path.split("/")[1:])
        target_document = None
        for document in self.engine.library.folders[folder].documents:
            if document.path == path:
                target_document = document
                break
        if not target_document:
            return f"{document_path} was not found in the library."

        return (
            f"{path}, page {page_number}:\n{target_document.read_page(page_number)}\n\n"
        )


class SearchAF(AgentFunction):
    """Searches into a Document.

    Args:
        AgentFunction (_type_): _description_
    """

    def __init__(self, name, description, engine) -> None:
        self.engine = engine
        super().__init__(name, description)

    def func(self, document_path: str, regular_expression: str) -> str:
        n_words_window = 10
        folder = document_path.split("/")[0]
        path = "/".join(document_path.split("/")[1:])
        target_document = None
        for document in self.engine.library.folders[folder].documents:
            if document.path == path:
                target_document = document
                break

        words = target_document.content.split()

        # Compile the regular expression
        pattern = re.compile(regular_expression)

        # Find all occurrences of the search expression
        matches = pattern.finditer(document.content)
        indices = [m.start() for m in matches]

        # Extract context around each occurrence
        contexts = []
        for index in indices:
            # Find the word index for the match
            word_index = sum(
                1 for _ in re.finditer(r"\b\w+\b", document.content[:index])
            )
            start = max(0, word_index - n_words_window)
            end = min(len(words), word_index + n_words_window + 1)
            context = " ".join(words[start:end])
            contexts.append(context)

        contexts_string = "Search results:\n"
        for context in contexts:
            contexts_string += f"[...] {context} [...]\n"
        return contexts_string
