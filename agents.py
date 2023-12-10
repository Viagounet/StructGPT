import inspect

import yaml

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


class JournalistAF(AgentFunction):
    def __init__(self, name, description, engine) -> None:
        self.engine = engine
        super().__init__(name, description)

    def func(self, subject: str, style: str, length: str, language: str) -> str:
        return self.engine.query(
            f"You're a highly skilled journalist. You're asked to write about :\nSubject: {subject}\nStyle: {style}\nLength: {length}\nLanguage: {language}",
            1024,
        ).content


class Agent:
    def __init__(self, engine, agent_functions) -> None:
        self.engine = engine
        self.agent_functions = agent_functions
        self.history = []

    @property
    def available_functions(self):
        function_descriptions = []
        for function in self.agent_functions:
            function_descriptions.append(function.usage)
        return "\n".join(function_descriptions)

    @property
    def header(self):
        return f"""Your goal is to [REPLACE].
To achieve this goal you will make good use of the following functions:
{self.available_functions}

Note: You will not make use of composite functions."""

    @property
    def body(self):
        if self.history == []:
            return ""
        else:
            return "\n".join(self.history)

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
                output = function.func(*args)
        return {
            "raw_answer": raw_gpt_answer,
            "explaination": explaination_string,
            "function_used": function,
            "arguments": args,
            "output": output,
        }


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

identical = AgentFunction("identical", "returns an identical string n times")
addition = AdditionAF("addition", "adds two number together")
journalist = JournalistAF(
    "journalist", "will write a news report with great skill about any subject", engine
)
my_agent = Agent(engine, [journalist, identical, addition])
agent_answer = my_agent.query("Write a report about how electricity works")
print(agent_answer)
