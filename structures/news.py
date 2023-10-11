from typing import List, Tuple
from structures.generator.generator import Generator

class ArticleAnalysis(Generator):
    title: str
    authors: List[str]
    date: str
    short_description: str
    summary: str
    questions_to_ask: List[str]
    def __init__(self, article_content, **kwargs):
        super().__init__(**kwargs)
        self.article_content = article_content

class ConflictAnalysis(Generator):
    beligerants: Tuple[str, str]
    beligerants_recent_news: Tuple[str, str]
    situation_summary: str
    win_percentages: Tuple[int, int]
    possible_future_events: List[str]

    def __init__(self, credible_source, **kwargs):
        super().__init__(**kwargs)
        self.credible_source = credible_source

