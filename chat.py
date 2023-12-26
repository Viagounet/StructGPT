import yaml

from engine import Engine
# from engine_mistral import MistralEngine

with open("examples/parameters/philosophy.yaml", "r") as file:
    parameters = yaml.safe_load(file)

engine = Engine("gpt-3.5-turbo", parameters)
# engine = MistralEngine(parameters, "You are a helpful agent eager to discuss with everyone.")

running = True
while running:
    user_input = input("You: ")
    if user_input == ":q":
        running = False
        break
    ans = engine.query(user_input)
    print("Engine: " + ans.content)