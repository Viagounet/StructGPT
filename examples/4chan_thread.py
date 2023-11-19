import json
from documents.document import Chunk, Text
from engine import Engine

with open("parameters.json", "r", encoding="utf-8") as f:
    parameters = json.load(f)

engine = Engine("gpt-3.5-turbo", parameters=parameters)
engine.library.create_folder("4chan threads")
engine.library.folders["4chan threads"].add_document("4chan:vt:61678494")

loving_documents = engine.find_similar_to("Gura is so cute. I love her more than anything", "4chan threads")

ans = engine.query_chunks("What do people love about Gura?", loving_documents)

ans = engine.query_chunks("Using the fans opinion, write a short 4channy message a Gura fan could be typing in a forum: ", loving_documents + [Text(f"Here is the analysis of Gura fans opinion on Gura: {ans.content}")])
print(ans)