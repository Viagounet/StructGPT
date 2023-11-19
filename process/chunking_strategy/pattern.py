import tiktoken

from documents.document import Chunk


def func(document, **kwargs):
    pattern = kwargs["pattern"]
    string = document.content
    return [
        Chunk(chunk_content, document, i)
        for i, chunk_content in enumerate(string.split(pattern))
    ]
