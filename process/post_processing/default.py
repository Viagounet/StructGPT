import re


def post_processing(string: str, **kwargs) -> str:
    clean_text = re.sub("\n(\w)\1*\n+", "\n", string)
    return clean_text.replace("\f", "")
