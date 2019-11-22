import os
import requests
import time
import pandas as pd
import config
from flask import request
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State

external_stylesheets = [
    "https://use.fontawesome.com/releases/v5.0.7/css/all.css",
    'https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css',
    'https://fonts.googleapis.com/css?family=Roboto&display=swap'
]

external_script = "https://raw.githubusercontent.com/MarwanDebbiche/post-tuto-deployment/master/src/dash/assets/gtag.js"

app = dash.Dash(
    __name__, 
    external_stylesheets=external_stylesheets,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
    suppress_callback_exceptions=True
)

app.scripts.append_script({
    "external_url": external_script
})

app.title = 'Reviews AI2Prod'

companies = pd.read_csv('./csv/companies_forbes.csv')
random_reviews = pd.read_csv('./csv/random_reviews.csv')

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

home_layout = html.Div(
    [
        html.Div(
            [
                html.A(
                    html.Img(
                        id='company_logo',
                        style={
                            'height': '100px',
                            'padding': '5px'
                        }
                    ),
                    id="company_link",
                    target="_blank"
                )
            ],
            style={
                'height': '100px',
                'backgroundColor': 'white',
                'borderStyle': 'solid',
                'borderRadius': '100px',
                'borderWidth': 'thin'
            }
        ),

        html.H1(
            [
                "What do you think of ",
                html.Span(
                    id='company_name'
                ),
                " ?"
            ],
            className="h3 mb-3 font-weight-normal",
            style={
                'marginTop': '5px'
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
            'Sentiment analysis 🤖'
        ),

        dbc.Progress(
            children=html.Span(
                id='proba',
                style={
                    'color': 'black',
                    'fontWeight': 'bold'
                }
            ),
            id="progress",
            striped=False,
            animated=False,
            style={
                'marginBottom': '10px'
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
            style={'marginBottom': '30px'}
        ),

        html.Button(
            [
                html.Span(
                    "Submit",
                    style={
                        "marginRight": "10px"
                    }
                ),
                html.I(
                    className="fa fa-paper-plane m-l-7"
                )
            ],
            className="btn btn-lg btn-primary btn-block",
            role="submit",
            id="submit_button",
            n_clicks_timestamp=0
        ),
        html.Button(
            [
                html.Span(
                    "Review another brand",
                    style={
                        "marginRight": "10px"
                    }
                ),
                html.I(
                    className="fas fa-sync-alt"
                )
            ],
            className="btn btn-lg btn-secondary btn-block",
            id='switch_button',
            n_clicks_timestamp=0
        ),
        html.P(
            dcc.Link("Go to Admin 🔑", id="admin-link", href="/admin"),
            className="mt-2"

        ),
        html.P(
            [
                html.A("BESBES", href="https://ahmedbesbes.com", target="_blank"),
                " / ",
                html.A("DEBBICHE", href="https://marwandebbiche.com",
                       target="_blank"),
                " - 2019"
            ],
            className="mt-3 mb-2 text-muted"
        ),
    ],
    className="form-review",
)

admin_layout = html.Div(
    [
        html.H1("Admin Page 🔑"),
        html.Div(id="admin-page-content"),
        html.P(
            dcc.Link("Go to Home 🏡", href="/"),
            style={"marginTop": "20px"}
        )
    ]
)


@app.callback(
    [
        Output('company_logo', 'src'),
        Output('company_name', 'children'),
        Output('review', 'value'),
        Output('company_link', 'href')
    ],
    [
        Input('submit_button', 'n_clicks_timestamp'),
        Input('switch_button', 'n_clicks_timestamp')
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
        Output('rating', 'value'),
        Output('submit_button', 'disabled')
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

        if proba >= 67:
            return text_proba, proba, 'success', suggested_rating, False
        elif 33 < proba < 67:
            return text_proba, proba, 'warning', suggested_rating, False
        elif proba <= 33:
            return text_proba, proba, 'danger', suggested_rating, False
    else:
        return None, 0, None, 0, True


# Load review table
@app.callback(
    Output('admin-page-content', 'children'),
    [Input('url', 'pathname')]
)
def load_review_table(pathname):
    if pathname != "/admin":
        return None

    response = requests.get(f"{config.API_URL}/reviews")

    reviews = pd.DataFrame(response.json())

    table = dbc.Table.from_dataframe(reviews,
                                     striped=True,
                                     bordered=True,
                                     hover=True,
                                     responsive=True,
                                     header=["id", "brand", "created_date", "review",
                                             "rating", "suggested_rating", "sentiment_score"],
                                     columns=["id", "brand", "created_date", "review",
                                              "rating", "suggested_rating", "sentiment_score"]
                                     )

    return table

# Update page layout


@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/':
        return home_layout
    if pathname == "/admin":
        return admin_layout
    else:
        return [
            html.Div(
                [
                    html.Img(
                        src="./assets/404.png",
                        style={
                            "width": "50%"
                        }
                    ),
                ],
                className="form-review"
            ),
            dcc.Link("Go to Home", href="/"),
        ]


if __name__ == '__main__':
    app.run_server(debug=config.DEBUG, host=config.HOST)
