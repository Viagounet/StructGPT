from agent_functions.generic import AgentFunction


class AdditionAF(AgentFunction):
    """Adds two numbers together.

    Args:
        AgentFunction (_type_): _description_
    """

    def __init__(self, name, description) -> None:
        super().__init__(name, description)

    def func(self, a: float, b: float) -> float:
        return a + b
