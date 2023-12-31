import dash
import dash_bootstrap_components as dbc
import yaml
from dash import html, dcc, Input, Output, State, MATCH, ALL
import json
from engine import Engine
#from engine_mistral import MistralEngine

with open("examples/parameters/philosophy.yaml", "r") as file:
    parameters = yaml.safe_load(file)

engine = Engine("gpt-3.5-turbo", parameters)

with open("../dataset_philo/0-2.json", "r", encoding="utf-8") as f:
    a = json.load(f)
prompt = a["prompt"].split("### Response:")[1]
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

def split_markdown(text):
    # Split the text by lines
    lines = text.split('\n')
    
    # Initialize variables
    parts = []
    current_part = []

    # Iterate through each line
    for line in lines:
        # Check if the line is a level 1 title
        if line.startswith('### '):
            # If there's content in the current part, add it to the parts list
            if current_part:
                parts.append('\n'.join(current_part))
                current_part = []
        # Add the line to the current part
        current_part.append(line)
    
    # Add the last part if it exists
    if current_part:
        parts.append('\n'.join(current_part))
    
    return parts

def create_text_div(raw_text):
    parts = split_markdown(raw_text)
    text_divs = []
    for i, part in enumerate(parts):
        text_divs.append(html.Div([dbc.Button("Extend", 
                                              style={"height":"70%"}, 
                                              color="secondary", id={"type":"expand-button", "index": i}), 
                                   dcc.Markdown(part, id={"type":"markdown-text", "index": i},
                                                style={"text-align":"justfy", "padding":"0.3rem"})], 
                                   className="d-flex flex-row gap-2"))
    return html.Div(text_divs, className="d-flex flex-column")

app.layout = html.Div([
    html.H1("StructGPT - Interface"),
    dbc.Textarea(placeholder="Write your query here", id="query-text-area"),
    dbc.Button("Send", id="send-button", className="mb-2"),
    html.Div(create_text_div(prompt), id="output", className="p-2")
], className="d-flex flex-column p-2")

@app.callback(
    Output("output", "children"),
    Input("send-button", "n_clicks"),
    State("query-text-area", "value"),
    prevent_initial_call=True
)
def update_markdown(n, query):
    if n:
        ans = engine.query(query, max_tokens=2048, temperature=0.2)
        return create_text_div(ans.content)
    return ""

@app.callback(
    Output({"type": "markdown-text", "index": MATCH}, "children"),
    Input({"type": "expand-button", "index": MATCH}, "n_clicks"),
    State({"type": "markdown-text", "index": MATCH}, "children"),
    prevent_initial_call=True
)
def expand_text(n, current_text):
    if n:
        ans = engine.query(f"Texte:\n{current_text}\nRéécrire le texte, en gardant le même style markdown et les mêmes titres principaux, mais en ajoutant de nouvelles parties, des exemples, des références et des pistes sur d'autres idées similaires.", max_tokens=2048, temperature=0)
        return ans.content
    return ""

if __name__ == "__main__":
    app.run_server()