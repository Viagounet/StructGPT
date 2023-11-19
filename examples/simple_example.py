from engine import Engine
import yaml

with open("parameters.yaml", "r") as file:
    parameters = yaml.safe_load(file)
    print(parameters)


engine = Engine("gpt-3.5-turbo", parameters=parameters)
engine.library.create_folder("mouse")
engine.library.folders["mouse"].add_document("test_data/cat.txt")

ans = engine.query_folder(
    "In the cat example, what kind of formating is used?",
    "mouse",
    100,
    top_N=5,
)
print(ans)
print(engine.print_logs_history())
