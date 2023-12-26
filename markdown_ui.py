import dash
import dash_bootstrap_components as dbc
import yaml
from dash import html, dcc, Input, Output, State

from engine import Engine
#from engine_mistral import MistralEngine

with open("examples/parameters/philosophy.yaml", "r") as file:
    parameters = yaml.safe_load(file)

engine = Engine("gpt-3.5-turbo", parameters)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H1("StructGPT - Interface"),
    dbc.Textarea(placeholder="Write your query here", id="query-text-area"),
    dbc.Button("Send", id="send-button"),
    dcc.Markdown(id="output")
], className="d-flex flex-column p-2")

@app.callback(
    Output("output", "children"),
    Input("send-button", "n_clicks"),
    State("query-text-area", "value")
)
def update_markdown(n, query):
    if n:
        ans = engine.query(query, max_tokens=512, temperature=0.2)
        return ans.content
    return ""
if __name__ == "__main__":
    app.run_server()