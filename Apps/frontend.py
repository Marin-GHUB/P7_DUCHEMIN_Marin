import re
from re import search

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from dash.development.base_component import Component
from numpy.core.fromnumeric import searchsorted

#################################################################
# Initialisation of variables
path_infos = f"/infos"
path_home = f"/"
result_text = ''
result_color = 'success'
client_id = dcc.Store(id='client_id', storage_type='local')
regex = re.compile('\?')

#################################################################
### Describing the entire Layout
# Styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# Padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

# Describing the sidebar
sidebar = html.Div(
    [
        html.H1("Navigation", className="display-5", style={'textAlign':'center'}),
        html.Hr(),
        html.P("Estimation d'Allocation de Prêt", className="lead", style={'textAlign':'center'}),
        dbc.Nav(
            [
                dbc.NavLink("Accueil", href=path_home, active="exact", id='home_path'),
                dbc.NavLink("Informations complémentaires", href=path_infos, active="exact", id='infos_path'),
                ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

# Describing the page content
content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

# Describing the meta layout
layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content
])

# Describing the home page
accueil = [
    dbc.Row(dbc.Col(html.H1('Prêt à Dépenser', style={'textAlign':'left'})), justify='center'),
    dbc.Row(html.Hr()),
    dbc.Row(html.Hr()),
    dbc.Row([
        dbc.Col(html.P("Merci de bien vouloir rentrer l'identifiant du client:"), width={'size':3}),
        dbc.Col(dbc.Input(id='client_input', placeholder='Enter ID', value=None), width={'size':2}),
        ], no_gutters=True),
    dbc.Row(html.Hr()),
    dbc.Row(dbc.Col(dbc.Label(result_text, id = 'result_text', color=result_color)))
]

# Describing the infos page
infos = [
    dbc.Row(dbc.Col(html.H1('Test Info Page'))),
    dbc.Row([
        dbc.Col(dbc.Label(client_id, id='shown_client_id')),
    ])
]

#################################################################
### Calling the app in Flask
def create_dash_application(flask_app):
    # Creating the app
    dash_app = dash.Dash(
        server=flask_app,
        name='front_end',
        url_base_pathname='/',
        external_stylesheets=[dbc.themes.SLATE]
    )
    
    # Display Layout
    dash_app.layout = layout

    # Processing page content
    @dash_app.callback(
        Output("page-content", "children"),
        [Input("url", "pathname")]
    )
    def render_page_content(pathname):
        if pathname == "/":
            return accueil
        elif pathname == "/infos":
            return infos
        # Creating an error 404 page
        else:
            return dbc.Jumbotron([
                    html.H1("404: Not found", className="text-danger"),
                    html.Hr(),
                    html.P(f"The pathname {pathname} was not recognised...")
                    ])

    # Storing the ID in the data storage
    @dash_app.callback(
        Output('client_id', 'data'),
        [Input('infos_path', 'href')]
    )
    def get_client_id(input_value):
        temp_list = regex.split(input_value)
        if len(temp_list) > 1 :
            ID_temp = temp_list[-1]
        else : 
            ID_temp = None
        return ID_temp
        
    return dash_app
    
#################################################################
#################################################################
