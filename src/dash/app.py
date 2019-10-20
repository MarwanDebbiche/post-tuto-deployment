# -*- coding: utf-8 -*-
import os
import requests

from random import choice
import pandas as pd

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

from flask import send_from_directory

external_stylesheets = ['https://maxcdn.bootstrapcdn.com/font-awesome/4.2.0/css/font-awesome.min.css',
                        'https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css',
                        'https://fonts.googleapis.com/css?family=Roboto&display=swap',
                        'https://raw.githubusercontent.com/ahmedbesbes/stylesheets/master/form-signin.css'
                        ]

app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,
                meta_tags=[
                    {"name": "viewport", "content": "width=device-width, initial-scale=1"}
                    ]
                )

companies = pd.read_csv('../data/scraping/companies.csv')

app.layout = html.Div(
    [
        html.Div(
            [
                html.Img(
                    id='company_logo',
                    style={
                        'width': '50%'
                    }
                ),
            ],
            style={
                'height': '150px'
            }
        ),

        html.H3(
            [
                html.Span(
                    id='company_name',
                    className="badge badge-secondary",
                    style={
                        'white-space': 'pre-wrap'
                    }
                )
            ]
        ),

        html.H1(
            "What do you think of this brand ?",
            className="h4 mb-3 font-weight-normal"
        ),

        html.Div(
            [
                dcc.Textarea(
                    className="form-control z-depth-1",
                    id="review",
                    rows="8",
                    placeholder="Write something here..."
                )
            ],
            className="form-group shadow-textarea"
        ),

        dbc.Progress(id="progress", striped=True, animated=True),

        html.Div(
            [
                html.H3(
                    id='proba'
                )
            ]
        ),
        html.Button(
            [
                html.Span(
                    "Submit",
                    style={
                        "margin-right": "10px"
                    }
                ),
                html.I(
                    className="fa fa-paper-plane m-l-7"
                )
            ],
            className="btn btn-lg btn-primary btn-block",
            role="submit"
        ),
        html.Button(
            [
                html.Span(
                    "Review another brand",
                    style={
                        "margin-right": "10px"
                    }
                ),
                html.I(
                    className="fa fa-refresh m-l-7"
                )
            ],
            className="btn btn-lg btn-secondary btn-block",
            id='another-brand',
            # role="submit"
        ),
        html.P("BESBES / DEBBICHE - 2019",
            className="mt-5 mb-3 text-muted"
        )

    ],
    className="form-signin",
)



@app.callback(
    [Output('company_logo', 'src'), Output('company_name', 'children'), Output('review', 'value')],
    [Input('another-brand', 'n_clicks')]
)
def change_brand(n_clicks):
    if n_clicks is not None:
        row = companies.sample(1).to_dict(orient="records")[0]
        random_logo = row['company_logo']
        random_name = row['company_name']
        if random_logo.startswith('http'):
            return random_logo, random_name, ''
        else:
            return 'https://' + random_logo, random_name, ''
    else:
        fnac_logo = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Fnac_Logo.svg/992px-Fnac_Logo.svg.png"
        return fnac_logo, "Fnac", ''


@app.callback(
    [Output('proba', 'children'), Output('progress', 'value')],
    [Input('review', 'value')]
)
def update_proba(review):
    if review is not None and review.strip() != '':
        response = requests.post(
            "http://localhost:5000/predict", data={'review': review})
        proba = response.json()

        return proba, round(proba * 100)
    else:
        return None, 0


if __name__ == '__main__':
    app.run_server(debug=True)
