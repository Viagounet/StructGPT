from agent_functions.commons.introspection import retrieve_function_details


class AgentFunction:
    """The generic agent class used to construct agent functions"""

    def __init__(self, name, description) -> None:
        self.name = name
        self.description = description

    @property
    def usage(self) -> str:
        """Returns a string that explains the usage of the functions (name, description, arguments)

        Returns:
            str: Function usage
        """
        return f"- {self.introspection()} ; {self.name} {self.description}"

    def func(self) -> None:
        """The actual function that will be executed. Note that when overloading this function, it is important to type your variables
        & choose clear names because it will be shown to GPT.
        """
        pass

    def introspection(self):
        """Returns the func() arguments and return types

        Returns:
            _type_: _description_
        """
        return retrieve_function_details(self, "func").replace("func", self.name)


class FinalAnswerAF(AgentFunction):
    """Function used to stop the agent

    Args:
        AgentFunction (AgentFunction): _description_
    """

    def __init__(self, name, description) -> None:
        super().__init__(name, description)

    def func(self, your_final_answer: str) -> str:
        return your_final_answer
