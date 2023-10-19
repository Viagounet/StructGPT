from ast import literal_eval
import os
from typing import get_type_hints
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
class Generator:
    def __init__(self, **kwargs):
        for arg, value in kwargs.items():
            setattr(self, arg, value)
    def as_json(self):
        return self.__dict__
    def source(self, user_prompt, mode="loose"):
        strict_message = ""
        if mode == "strict":
            strict_message = "- Do NOT invent values. If you don't know, write 'None'."
        if mode == "creative":
            strict_message = "- You are allowed in invent values and must fill all the blanks."
        hints = get_type_hints(self)
        full_prompt = f"""Class information
---
Class name: {type(self).__name__}
Class 'known values': {self.__dict__.items()}
Class 'to fill values' : {hints}

---

Task
---
- Answer the task by filling the arguments of {type(self).__name__} respecting hints type, in a JSON format. 
- Do not return the 'known values'. 
- DO not in any occasion change the values of 'known values'.
{strict_message}

The user asked: {user_prompt}
---

>{{"""
        output = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": full_prompt}],
                    max_tokens=1024,
                    temperature=1
                )["choices"][0]["message"]["content"]
        output = "{" + output
        if output[-1] != "}":
            output = output + "}"
        out = literal_eval(output)
        for key, value in self.__dict__.items():
            out[key] = value
        self.__init__(**out)
        return self
    def act(self, attribute, action):
        attribute_value = getattr(self, attribute)
        prompt = f"""{attribute_value} -> {action}"""
        output = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=1024,
            temperature=1
        )["choices"][0]["message"]["content"]
        setattr(self, attribute, output)
        return self
    def __str__(self) -> str:
        return str(self.as_json())