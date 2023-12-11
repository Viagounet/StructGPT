from agent_functions.generic import AgentFunction


class JournalistAF(AgentFunction):
    """Roleplays as a journalist using GPT.

    Args:
        AgentFunction (AgentFunction): _description_
    """

    def __init__(self, name, description, engine) -> None:
        self.engine = engine
        super().__init__(name, description)

    def func(self, subject: str, style: str, length: str, language: str) -> str:
        return self.engine.query(
            f"You're a highly skilled journalist. You're asked to write about :\nSubject: {subject}\nStyle: {style}\nLength: {length}\nLanguage: {language}",
            1024,
        ).content
