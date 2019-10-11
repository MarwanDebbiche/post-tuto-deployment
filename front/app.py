import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State


app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Review Analyzer")
])


if __name__ == '__main__':
    app.run_server(
        debug=True,
        host='0.0.0.0'
    )
