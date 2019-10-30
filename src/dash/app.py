import os
import requests
import pandas as pd
import config
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

external_stylesheets = [
    "https://use.fontawesome.com/releases/v5.0.7/css/all.css",
    'https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css',
    'https://fonts.googleapis.com/css?family=Roboto&display=swap',
    'https://raw.githubusercontent.com/ahmedbesbes/stylesheets/master/form-signin.css'
]

app = dash.Dash(
    __name__, external_stylesheets=external_stylesheets,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

companies = pd.read_csv('./csv/companies.csv')

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

        dbc.Button(
            children=[
                html.Span(
                    id='company_name'
                ),
                html.I(
                    className="fas fa-link",
                    style={
                        'margin-left': '5px'
                    }
                )
            ],
            id='button_company',
            color="primary",
            className="mr-1",
            outline=False,
            style={
                'margin-top': '10px'
            }
        ),
        
        html.H1(
            "What do you think of this brand ?",
            className="h4 mb-3 font-weight-normal",
            style={
                'margin-top': '5px'
            }
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

        dbc.Progress(
            children=html.Span(
                id='proba',
                style={
                    'color': 'black',
                    'font-weight': 'bold'
                }
            ),
            id="progress",
            striped=False,
            animated=False
        ),

        html.Hr(),

        html.Div(
            [
                dcc.Slider(
                    id='suggested_rating',
                    max=100
                ),
            ],
            style={'margin-bottom': '5px'}
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
                    className="fas fa-sync-alt"
                )
            ],
            className="btn btn-lg btn-secondary btn-block",
            id='another-brand',
        ),
        html.P(
            "BESBES / DEBBICHE - 2019",
            className="mt-5 mb-3 text-muted"
        )
    ],
    className="form-signin",
)


@app.callback(
    [
        Output('company_logo', 'src'),
        Output('company_name', 'children'),
        Output('review', 'value'),
        Output('button_company', 'href')
    ],
    [Input('another-brand', 'n_clicks')]
)
def change_brand(n_clicks):
    if n_clicks is not None:
        row = companies.sample(1).to_dict(orient="records")[0]

        random_logo = row['company_logo']
        if not random_logo.startswith('http'):
            random_logo = 'https://' + random_logo

        random_name = row['company_name']
        random_website = row['company_website']

        return random_logo, random_name, '', random_website
    else:
        fnac_logo = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Fnac_Logo.svg/992px-Fnac_Logo.svg.png"
        fnac_name = 'Fnac'
        return fnac_logo, fnac_name, '', 'http://google.com'


@app.callback(
    [
        Output('proba', 'children'),
        Output('progress', 'value'),
        Output('progress', 'color'),
        Output('suggested_rating', 'value')
    ],
    [Input('review', 'value')]
)
def update_proba(review):
    if review is not None and review.strip() != '':
        response = requests.post(
            "http://localhost:5000/api/predict", data={'review': review})
        proba = response.json()
        proba = round(proba * 100, 2)
        text_proba = f"{proba}%"

        if 60 < proba < 100:
            suggested_rating = proba
            return text_proba, proba, 'success', suggested_rating
        elif 40 < proba < 60:
            suggested_rating = proba
            return text_proba, proba, 'warning', suggested_rating
        elif proba < 40:
            suggested_rating = proba
            return text_proba, proba, 'danger', suggested_rating
    else:
        return None, 0, None, 50


if __name__ == '__main__':
    app.run_server(debug=True, host=config.HOST)
