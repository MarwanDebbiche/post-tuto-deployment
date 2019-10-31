import os
import requests
import pandas as pd
import config
from flask import request
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State

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

        html.H5(
            'Sentiment anlysis 🤖'
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
            animated=False,
            style={
                'margin-bottom': '10px'
            }
        ),

        html.H5(
            'Propose a rating 😁📢'
        ),

        html.Div(
            [
                dcc.Slider(
                    id='rating',
                    max=5,
                    min=1,
                    step=1,
                    marks={i: f'{i}' for i in range(1, 6)}
                ),
            ],
            style={'margin-bottom': '30px'}
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
            role="submit",
            id="submit-button",
            n_clicks_timestamp=0
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
            n_clicks_timestamp=0
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
    [
        Input('submit-button', 'n_clicks_timestamp'),
        Input('another-brand', 'n_clicks_timestamp')
    ],
    [
        State('review', 'value'),
        State('progress', 'value'),
        State('rating', 'value'),
        State('company_name', 'children')
    ]
)
def change_brand(submit_click_ts, another_brand_click_ts, review_text, score, rating, brand_name):
    if submit_click_ts > another_brand_click_ts:
        sentiment_score = float(score) / 100
        ip_address = request.remote_addr
        user_agent = request.headers.get('User-Agent')
        response = requests.post(
            f"{config.API_URL}/review",
            data={
                'review': review_text,
                'rating': rating,
                'suggested_rating': min(int(sentiment_score * 5 + 1), 5),
                'sentiment_score': sentiment_score,
                'brand': brand_name,
                'user_agent': user_agent,
                'ip_address': ip_address
            }
        )

        if response.ok:
            print("Review Saved")
        else:
            print("Error Saving Review")

    random_company = companies.sample(1).to_dict(orient="records")[0]

    company_logo_url = random_company['company_logo']
    if not company_logo_url.startswith('http'):
        company_logo_url = 'https://' + company_logo_url

    company_name = random_company['company_name']
    company_website = random_company['company_website']

    return company_logo_url, company_name, '', company_website


@app.callback(
    [
        Output('proba', 'children'),
        Output('progress', 'value'),
        Output('progress', 'color'),
        Output('rating', 'value')
    ],
    [Input('review', 'value')]
)
def update_proba(review):
    if review is not None and review.strip() != '':
        response = requests.post(
            f"{config.API_URL}/predict", data={'review': review})
        proba = response.json()
        proba = round(proba * 100, 2)
        suggested_rating = min(int((proba / 100) * 5 + 1), 5)
        text_proba = f"{proba}%"

        if suggested_rating >= 4:
            return text_proba, proba, 'success', suggested_rating
        elif 2 < suggested_rating < 4:
            return text_proba, proba, 'warning', suggested_rating
        elif suggested_rating <= 2:
            return text_proba, proba, 'danger', suggested_rating
    else:
        return None, 0, None, 0


if __name__ == '__main__':
    app.run_server(debug=config.DEBUG, host=config.HOST)
