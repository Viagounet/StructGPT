from datetime import datetime
import os
import openai

from sentence_transformers import SentenceTransformer
from documents.document import Prompt, Text
from documents.library import Library


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
        self.parameters = {
            "chunking_strategy": {
                ".txt": {"strategy": "pattern", "pattern": "\n"},
                ".py": {"strategy": "pattern", "pattern": "class"},
            }
        }
        self.library = Library(chunking_strategy=self.parameters["chunking_strategy"])

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
        return answer

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
