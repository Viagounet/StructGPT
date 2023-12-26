import datetime
import json
import os
from typing import List
import openai

from sentence_transformers import SentenceTransformer
from documents.document import Library, Text, Chunk, Prompt


class AnswerLog:
    def __init__(self, prompt, answer, model, prices_prompt, prices_completion):
        now = datetime.datetime.now()
        self.date = now.strftime("%d/%m/%Y %H:%M:%S")
        self.model = model
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

    @property
    def data(self):
        return {
            "date": self.date,
            "model": self.model,
            "prompt": self.prompt.content,
            "answer": self.answer.content,
            "prices": self.prices,
        }


class Engine:
    def __init__(self, model, parameters=None):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model

        now = datetime.datetime.now()

        self.run_id = "run_" + now.strftime("%Y%m%d_%H%M%S")
        self.n_queries = 0

        self.embeddings_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.logs = []
        self.prices_prompt = {
            "gpt-3.5-turbo": 0.002 / 1000,
            "gpt-4": 0.03 / 1000,
            "gpt-4-32k": 0.06 / 1000,
            "gpt-4-1106-preview": 0.01 / 1000
        }
        self.prices_completion = {
            "gpt-3.5-turbo": 0.002 / 1000,
            "gpt-4": 0.06 / 1000,
            "gpt-4-32k": 0.12 / 1000,
            "gpt-4-1106-preview": 0.03 / 1000
        }
        self.parameters = parameters
        if self.parameters["logs"]["autosave"]:
            os.mkdir(f"{self.parameters['logs']['save_path']}/{self.run_id}")
        self.library = Library(self.parameters)

    def find_similar_to(self, example_verbatim: str, folder: str, N=10):
        example_verbatim = Text(example_verbatim)
        example_verbatim.create_embeddings(self.embeddings_model)
        folder = self.library.folders[folder]
        folder.create_embeddings(self.embeddings_model)
        return example_verbatim.top_N_similar(folder.documents, N)

    def query(self, prompt: str, max_tokens: int = 256, temperature: float = 0):
        self.n_queries += 1
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
        if self.parameters["logs"]["autosave"]:
            with open(
                f"{self.parameters['logs']['save_path']}/{self.run_id}/{self.n_queries}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(answer_log.data, f, ensure_ascii=False, indent=4)
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
            documents = prompt.top_N_similar(documents, N=top_N)
        for document in documents:
            prompt.add_document(document)
        return self.query(prompt.content, max_tokens=max_tokens)

    def query_chunks(
        self,
        prompt_initial: str,
        chunks: List[Chunk],
        max_tokens: int = 256,
        temperature: float = 0,
    ):
        prompt_content = prompt_initial
        prompt = Prompt(prompt_content)
        prompt.create_embeddings(self.embeddings_model)
        for chunk in chunks:
            chunk.create_embeddings(self.embeddings_model)
        for chunk in chunks:
            prompt.add_chunk(chunk)
        print("long")
        print(prompt.tokens)
        if prompt.tokens > 512:
            if (
                self.parameters["engine"]["too_many_tokens_strategy"]
                == "summarize_chunks"
            ):
                prompt.reset()
                for chunk in chunks:
                    print("Summ..")
                    chunk.short_content = self.query(
                        f"List the important information, keep only the essential : {chunk.content}"
                    ).content
                for chunk in chunks:
                    prompt.add_chunk(chunk, short=True)
        print("short")
        print(prompt.tokens)
        return self.query(prompt.content, max_tokens=max_tokens)

    def print_logs_history(self):
        logs_string = ""
        for log in self.logs:
            logs_string += f"- {log.date}\n\t- Prompt: {log.prompt.content}\n\t- \n\t- Answer: {log.answer.content}\n\t- Price: {round(log.prices['total'], 5)}$ (prompt -> {round(log.prices['prompt'], 5)} / completion -> {round(log.prices['completion'], 5)})\n\n---\n"
        print(logs_string)
