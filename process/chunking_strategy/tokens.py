import tiktoken

from documents.document import Chunk


def func(document, **kwargs):
    pattern = kwargs["pattern"]
    max_tokens = kwargs["max_tokens"]
    string = document.content
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
        Chunk(chunk_content, document, i)
        for i, chunk_content in enumerate(transcript_parts)
    ]
