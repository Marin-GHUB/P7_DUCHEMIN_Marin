import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash.dependencies import Input, Output

#################################################################
# Initialisation of variables
client_result = ''
result_text = ''

#################################################################
# Describing the Layout
layout = html.Div([
    dbc.Row(dbc.Col(html.H1('This is a test'))),
    dbc.Row([
        dbc.Col(html.P('Enter the ID you want :')),
        dbc.Col(dbc.Input(id='client_id', placeholder='Enter ID')),
        dbc.Col(dbc.Label(result_text, id = 'result_text')),
        ]) 
])

#################################################################
# Calling the app in Flask
def create_dash_application(flask_app):
    dash_app = dash.Dash(
        server=flask_app,
        name='front_end',
        url_base_pathname='/',
        external_stylesheets=[dbc.themes.SLATE]
    )

    dash_app.layout = layout

    return dash_app
    
#################################################################
#################################################################