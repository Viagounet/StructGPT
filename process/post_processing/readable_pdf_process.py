import re


def post_processing(string: str, **kwargs) -> str:
    pattern = r"(?<=[a-z])\n(?=[a-z\s])"
    clean_text = re.sub(pattern, "", string)
    return clean_text.replace("\f", "")
