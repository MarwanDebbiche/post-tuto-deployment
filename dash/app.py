import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import requests
import os


API_URL = os.getenv('API_URL')
if API_URL is None:
    API_URL = "http://localhost:5000/api"

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Review Analyzer"),
    dcc.Textarea("review-input"),
    html.Button(id='submit-button', n_clicks=0, children='Submit'),
    html.Div(id="submit-output", children="")
])


@app.callback(
    Output('submit-output', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('review-input', 'value')]
)
def sumbit_review_and_predict(n_clicks, review):
    if not n_clicks:
        return ""

    elif review:
        r = requests.post(url=f"{API_URL}/predict-rating", data={"review": review})
        rating = r.json()
        return [
            html.Div("Thank you for the feedback."),
            html.Div(f"Suggested grade : {rating}")
         ]
    else:
        return "Please enter a review"


if __name__ == '__main__':
    app.run_server(
        debug=True,
        host='0.0.0.0'
    )
