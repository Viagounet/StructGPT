import json
from engine import Engine

with open("new_parameters_format.json", "r", encoding="utf-8") as f:
    parameters = json.load(f)

engine = Engine("gpt-3.5-turbo", parameters=parameters)

google_API = parameters["variables"]["services"]["gsearch"]["API"]
google_CSE = parameters["variables"]["services"]["gsearch"]["CSE"]

engine.library.create_folder("Web Search")
pages = engine.library.web_search(
    "Ukraine-Russia latest news", google_API, google_CSE, skip_files=True
)
for page in pages:
    engine.library.folders["Web Search"].add_document(page)
ans = engine.query_folder("Ukraine-Russia latest news", "Web Search", 512, top_N=5)
print(ans)
