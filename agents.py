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
        return string * 5

    def introspection(self):
        return retrieve_function_details(self, "func").replace("func", self.name)


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
{self.available_functions}"""

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
        return self.engine.query(prompt, 512)


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


with open("examples/parameters/philosophy.yaml", "r") as file:
    parameters = yaml.safe_load(file)

engine = Engine("gpt-4", parameters=parameters)

identical = AgentFunction("identical", "returns an identical string n times")
my_agent = Agent(engine, [identical])
agent_answer = my_agent.query("Repeat 'I love Maria' 6 times")
print(agent_answer)
