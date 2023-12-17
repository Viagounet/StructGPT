import inspect

import yaml
from documents.document import Text

from engine import Engine


class AgentFunction:
    def __init__(self, name, description) -> None:
        self.name = name
        self.description = description

    @property
    def usage(self):
        return f"- {self.introspection()} ; {self.name} {self.description}"

    def func(self, string: str, n_times: int) -> str:
        return string * n_times

    def introspection(self):
        return retrieve_function_details(self, "func").replace("func", self.name)


class AdditionAF(AgentFunction):
    def __init__(self, name, description) -> None:
        super().__init__(name, description)

    def func(self, a: float, b: float) -> float:
        return a + b


class FinalAnswerAF(AgentFunction):
    def __init__(self, name, description) -> None:
        super().__init__(name, description)

    def func(self, your_final_answer: str) -> str:
        return your_final_answer


class JournalistAF(AgentFunction):
    def __init__(self, name, description, engine) -> None:
        self.engine = engine
        super().__init__(name, description)

    def func(self, subject: str, style: str, length: str, language: str) -> str:
        return self.engine.query(
            f"You're a highly skilled journalist. You're asked to write about :\nSubject: {subject}\nStyle: {style}\nLength: {length}\nLanguage: {language}",
            1024,
        ).content


class DocumentMetaDataAF(AgentFunction):
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
        return document.chunks[chunk_number].content


class Agent:
    def __init__(self, engine, agent_functions) -> None:
        self.engine = engine
        self.agent_functions = agent_functions
        self.history = []
        self.history_summary = ""

    @property
    def available_functions(self):
        function_descriptions = []
        for function in self.agent_functions:
            function_descriptions.append(function.usage)
        return "\n".join(function_descriptions)

    @property
    def display_documents(self):
        documents_string = "The following are the files you can work with. Always write their full path.\n"
        for folder in self.engine.library.folders:
            for document in self.engine.library.folders[folder].documents:
                documents_string += f"- {folder}/{document.path}"
        return documents_string

    @property
    def header(self):
        return f"""Your goal is to [REPLACE].
To achieve this goal you will make good use of the following functions:
{self.available_functions}

Note: You will not make use of composite functions."""

    @property
    def history_text(self):
        history_string = "These were the previous actions & results :\n\n"
        for i, answer in enumerate(self.history):
            func_name = answer["function_used"].name
            args = answer["arguments"]
            output = answer["output"]
            history_string += (
                f"- Action {i+1}: {func_name}\nArguments: {args}\nOutput: {output}\n"
            )
        return Text(history_string)

    def summarize_history(self, goal):
        print("==========================================================")
        ans = engine.query(
            f"The user goal was to : '{goal}'. Here is the history of your actions to accomplish the user goal :\n{self.history_summary}\n{self.history_text.content}\n---\nIn a bullet point format you will summarize : what you've learn in relation to the user's request & what is left to answer to fullfill the request",
            max_tokens=2048,
        )
        print(ans.content)
        self.history_summary = (
            f"Here is a summary of what happened before: {ans.content}"
        )
        print(
            "================================================================================"
        )
        self.history = []

    @property
    def body(self):
        if self.history == [] and self.summarize_history == "":
            return {self.display_documents}
        elif self.history == [] and self.summarize_history != "":
            return f"{self.display_documents}\n\n{self.history_summary}"
        else:
            return f"{self.display_documents}\n\n{self.history_summary}\n\n{self.history_text.content}"

    @property
    def footer(self):
        return "---\nYou will now answer with an action (using a function) by precisely following this template :\n\nExplaination: Replace this text with your reasoning behind your action choice.\nAction: function(argument1, ...)"

    @property
    def full_prompt(self):
        return f"{self.header}\n{self.body}\n{self.footer}"

    def query(self, instruction):
        prompt = self.full_prompt.replace("[REPLACE]", instruction)
        raw_gpt_answer = self.engine.query(prompt, 512)
        explaination_string = raw_gpt_answer.content.split("Action: ")[0]
        function_string = raw_gpt_answer.content.split("Action: ")[1]
        func_name, args = parse_function_string(function_string)
        output = "No output"
        for function in self.agent_functions:
            if function.name == func_name:
                try:
                    output = function.func(*args)  # Actually executes the function
                except Exception as e:
                    output = f"An error has occured: {e}"
                break
        answer = {
            "raw_answer": raw_gpt_answer,
            "explaination": explaination_string,
            "function_used": function,
            "arguments": args,
            "output": output,
        }
        self.history.append(answer)
        return answer

    def run(self, instructions):
        stop = False
        while not stop:
            print(">>>>", self.history_text.n_tokens)
            if self.history_text.n_tokens > 2000:
                self.summarize_history(instructions)
            ans = self.query(instructions)
            if ans["function_used"].name == "final_answer":
                stop = True
            else:
                print("(not yet completed)", ans["output"])
        return print("\n(completed)", ans["output"])

    def save_history(self, save_path: str):
        file = open(save_path, "w", encoding="utf-8")
        for ans in self.history:
            file.write(
                f"Explaination: {ans['explaination']}\nFunction used: {ans['function_used']}\nOutput: {ans['output']}\n-------\n"
            )
        file.close()


def retrieve_function_details(cls, method_name):
    method = getattr(cls, method_name, None)
    if method is None:
        return "Method not found"

    sig = inspect.signature(method)
    params = sig.parameters
    return_annotation = sig.return_annotation

    # Formatting the arguments and their types, excluding 'self'
    arg_list = [
        f"{name}: {param.annotation.__name__}"
        for name, param in params.items()
        if name != "self"
    ]
    args = ", ".join(arg_list)

    # Formatting the return type
    ret_type = (
        return_annotation.__name__
        if return_annotation is not inspect._empty
        else "None"
    )

    return f"{method_name}({args}) -> {ret_type}"


import re
import ast


def parse_function_string(func_str):
    # Regular expression pattern to match the function call structure
    pattern = r"(\w+)\((.*)\)"
    match = re.match(pattern, func_str)

    if match:
        func_name = match.group(1)
        args_str = match.group(2)

        try:
            # Using ast.literal_eval to safely evaluate the argument string
            args = ast.literal_eval(f"[{args_str}]")
        except Exception as e:
            return f"Error parsing arguments: {e}", []

        return func_name, args
    else:
        return "Invalid function string", []


with open("examples/parameters/philosophy.yaml", "r") as file:
    parameters = yaml.safe_load(file)

engine = Engine("gpt-4", parameters=parameters)
engine.library.create_folder("documents")
engine.library.folders["documents"].add_document("test_data/pfe.txt")


identical = AgentFunction("identical", "returns an identical string n times")
addition = AdditionAF("addition", "adds two number together")
final_answer = FinalAnswerAF("final_answer", "your final answer to the user")
document_reader = ReadDocumentAF(
    "read_document", "will return the content of the document", engine
)
chunk_reader = ReadChunkAF(
    "read_chunk",
    "will return the content of a document chunk (index starts at 0)",
    engine,
)

journalist = JournalistAF(
    "journalist", "will write a news report with great skill about any subject", engine
)
metadata = DocumentMetaDataAF(
    "metadata",
    "returns metadata about the document (type, number of pages, chunks, letters etc.)",
    engine,
)

my_agent = Agent(
    engine,
    [
        final_answer,
        journalist,
        identical,
        addition,
        metadata,
        document_reader,
        chunk_reader,
    ],
)


instructions = """I want you to summarize what this document is and what it talks about.
More precisely, I want you to tell me the overall structure of the document (what are the main parts? what language? what are the motivations? who's the author?)
Then I want you to tell me more about the document content and to give example of similar documents that could be useful as references.
Finally, I want you to propose improvements on the document. Please give a rather in-depth answer."""

agent_answer = my_agent.run(instructions)
my_agent.save_history("agent_history.txt")
