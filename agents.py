from agent_functions.commons.introspection import parse_function_string
from agent_functions.generic import FinalAnswerAF


class Agent:
    def __init__(self, engine, agent_functions) -> None:
        self.engine = engine
        self.final_answer = FinalAnswerAF(
            "final_answer", "your final answer to the user"
        )
        self.agent_functions = [self.final_answer] + agent_functions
        self.history = []
        self.remaining_actions = 5

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
    def body(self):
        history_string = ""
        if self.history == []:
            return {self.display_documents}
        else:
            history_string += "These were the previous actions & results :\n\n"
            for i, answer in enumerate(self.history):
                func_name = answer["function_used"].name
                args = answer["arguments"]
                output = answer["output"]
                history_string += f"- Action {i+1}: {func_name}\nArguments: {args}\nOutput: {output}\n"
            return f"{self.display_documents}\n\n{history_string}"

    @property
    def footer(self):
        if self.remaining_actions:
            return f"---\nYou have {self.remaining_actions} actions left. You will now answer with an action (using a function) by precisely following this template :\n\nExplaination: Replace this text with your reasoning behind your action choice.\nAction: function(argument1, ...)"
        return "---\nYou will now answer with your FINAL action following this template:\n\nExplaination: Replace this text with your reasoning behind your action choice.\nAction: function(argument1, ...)"

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
        self.remaining_actions -= 1
        if self.remaining_actions == 0:
            self.agent_functions = [self.final_answer]
        return answer

    def run(self, instructions: str, max_actions: int):
        self.remaining_actions = max_actions
        stop = False
        while not stop:
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
                f"Explaination: {ans['explaination']}\nFunction used: {ans['function_used'].name}\nArguments: {ans['arguments']}\nOutput: {ans['output']}\n-------\n"
            )
        file.close()
