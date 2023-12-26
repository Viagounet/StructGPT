import yaml

#from engine import Engine
from engine_mistral import MistralEngine

with open("examples/parameters/philosophy.yaml", "r") as file:
    parameters = yaml.safe_load(file)

#engine = Engine("gpt-3.5-turbo", parameters)
engine = MistralEngine("./mistral_philosopher-fr", parameters, "Vous êtes un professeur de philosophie, et vous répondez de manière détaillée aux questions d'un élève.", cpu_offload=False)

running = True
while running:
    user_input = input("You: ")
    if user_input == ":q":
        running = False
        break
    ans = engine.query(user_input, max_tokens=2048)
    print("Engine: " + ans.content)