# -*- coding: utf-8 -*-
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from flask import send_from_directory

external_stylesheets = ['https://maxcdn.bootstrapcdn.com/font-awesome/4.2.0/css/font-awesome.min.css',
                        'https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css',
                        'https://fonts.googleapis.com/css?family=Roboto&display=swap',
                        'https://raw.githubusercontent.com/ahmedbesbes/stylesheets/master/form-signin.css'
                        ]

app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets
                )

app.layout = html.Form(
                    [
                        html.Img(
                            src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Fnac_Logo.svg/992px-Fnac_Logo.svg.png",
                            style={
                                'width': '150px',
                                'height': '150px'
                            }
                        ),
                        html.H1(
                            "What do you think of this brand ?",
                            className="h4 mb-3 font-weight-normal"

                        ),
                        html.Div(
                            [
                                html.Textarea(
                                    className="form-control z-depth-1",
                                    id="exampleFormControlTextarea6",
                                    rows="8",
                                    placeholder="Write something here..."
                                )
                            ],
                            className="form-group shadow-textarea"
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
                            role="submit"
                        ),
                        html.P(
                            "© BESBES / DEBBICHE - 2019",
                            className="mt-5 mb-3 text-muted"
                        )

                    ],
                    className="form-signin",
                    )


if __name__ == '__main__':
    app.run_server(debug=True)
