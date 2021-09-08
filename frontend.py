#################################################################
### Import libraries ###

import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import pandas as pd
import plotly.graph_objs as go
import requests
from dash.dependencies import Input, Output, State

#################################################################
### Initilizations ###

# App
frontend = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE],
                    suppress_callback_exceptions=True)
server = frontend.server

#################################################################
### Designing the layout ###

# Describing the page content
content = html.Div([
    dbc.Row([
        dbc.Col(html.H1('Prêt à Dépenser', style={'textAlign':'center'})),
        dbc.Col(dbc.Label(''), width={'size':5})
    ]),
    dbc.Row(html.Hr()),
    dbc.Row(html.Hr()),
    dbc.Row([
        dbc.Col(html.Hr(), width={'size':2}),
        dbc.Col(html.P("Identifiant du client:"), width={'size':1}),
        dbc.Col(dbc.Input(
            id='client_input', placeholder='exemple : 100106'), width={'size':2}
        ),
        dbc.Col(dbc.Label('', id = 'check_input', color='light'), width={'size':2}),
        ]),
    dbc.Row(html.Hr()),
    dbc.Row([
        dbc.Col(html.Hr(), width={'size':1}),
        dbc.Col([
            dbc.Button("Résultat", id='result_button', n_clicks=0, block=True),
            dbc.Collapse(
                dbc.Card(dbc.CardBody(dbc.Label('', id='result_text', color='light'))),
                id='collapse_result',
                is_open=False,
            ),
        ], width={'size':3}),
        dbc.Col([
            dbc.Button("Certitude", id='proba_button', n_clicks=0, block=True),
            dbc.Collapse(
                dbc.Card(dbc.CardBody(dbc.Label('', id='proba_text', color='light'))),
                id='collapse_proba',
                is_open=False,
            ),
        ], width={'size':3}),        
        ]),
    dbc.Row(html.Hr()),
    dbc.Row([
        dbc.Col(html.Hr(), width={'size':1}),
        dbc.Col([
            dbc.Button("Informations Complémentaire", id='info_button', n_clicks=0, block=True),
            dbc.Row(html.Hr()),
            dbc.Fade([
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Graphique des Attributs Essentiels pour l'Obtient d'un Prêt"),
                        dbc.CardBody([
                            html.Div([
                                dbc.Row([
                                    dbc.Col(dcc.Graph(id='graph_a'), width={'size':12}),
                                ]),
                                dbc.Row([
                                    dbc.Col(html.Hr(), width={'size':2}),
                                    dbc.Col(dbc.Label("Les scores du client doivent être suppérieurs aux scores de référence.", color='grey')),
                                ]),
                                dbc.Row([
                                    dbc.Col(html.Hr(), width={'size':2}),
                                    dbc.Col(dbc.Label("Cliquez sur un des points pour afficher plus d'informations.", color='grey')),
                                ]),
                            ]),
                            
                        ])
                    ])),
                ]),
                dbc.Fade([
                    dbc.Row(
                        dbc.Col(dbc.Card([
                            dbc.CardHeader("", id='graph_b_title'),
                            dbc.CardBody([
                                html.Div([
                                    dbc.Row([
                                        dbc.Col(dcc.Graph(id='graph_b'), width={'size':12}),                                            
                                    ]),
                                    dbc.Row([
                                        dbc.Col(dbc.Label("", id='test_graph', color='grey')),
                                    ]),
                                ])
                            ])
                        ])),
                    )],
                    id='collapse_graph_b',
                    is_in=False,
                    appear=False
                )],
                id='collapse_info',
                is_in=False,
                appear=False
            ),
        ], width={'size':6})
    ]),
],id="page_content")

# Describing the meta layout
layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id='client_id'),
    dcc.Store(id='client_proba'),
    dcc.Store(id='client_score'),
    content,
])

#################################################################
### Deploying the layout ###

### Displaying the layout
frontend.layout = layout

### Managing collapses
# Collapse result
@frontend.callback(
    Output("collapse_result", "is_open"),
    [Input("result_button", "n_clicks")],
    [State("collapse_result", "is_open")],
)
def toggle_collapse_result(n, is_open):
    if n:
        return not is_open
    return is_open

# Collapse certitude
@frontend.callback(
    Output("collapse_proba", "is_open"),
    [Input("proba_button", "n_clicks")],
    [State("collapse_proba", "is_open")],
)
def toggle_collapse_proba(n, is_open):
    if n:
        return not is_open
    return is_open

# Collapse infos
@frontend.callback(
    Output("collapse_info", "is_in"),
    [Input("info_button", "n_clicks")],
    [State("collapse_info", "is_in")],
)
def toggle_fade_collapse(n, is_in):
    if n:
        return not is_in
    return is_in

#################################################################
### Getting the dataframe from the backend ###

backend_address = 'https://p7-backend.herokuapp.com'

# Function

## Due to memory limitation on Heroku, we will not look for the dataframe on the backend server ##
#def getting_df():
#    question = requests.get(f'{backend_address}/dataframe')
#    result = question.json()
#    return result
#
# Creating Variables
#json_df = getting_df()
#client_df = pd.read_json(json_df, orient='records')

# Creating Variables
client_df = pd.read_csv('Ressources/application_test.csv')
del json_df
accepted_clients_df = client_df[client_df['RESULT'] == 1]
refused_clients_df = client_df[client_df['RESULT'] == 0]
col_list = ['EXT_SOURCE_MIN', 'AMT_ANNUITY', 'CREDIT_TO_ANNUITY_RATIO','DAYS_BIRTH']
graph_col = ['Score Extérieur', 'Fond de Retraite', 'Ratio Credit / Fond de Retraite', "Score d'Âge"]
graph_dict = dict(zip(graph_col, col_list))

### Backend Callbacks ###

# Storing the ID in the data storage
def storing_id(ID):
    question = requests.post(f'{backend_address}/id', json={'id' : ID})
    question.json()

# Prediction
def predict_score():
    question = requests.get(f'{backend_address}/predict')
    result = question.json()
    return result

# Getting the ID
def getting_id():
    question = requests.get(f'{backend_address}/id')
    result = question.json()
    return result

# Getting the proba
def getting_proba():
    question = requests.get(f'{backend_address}/proba')
    result = question.json()
    return result

# Getting the score
def getting_score():
    question = requests.get(f'{backend_address}/score')
    result = question.json()
    return result

# Looking for the mean of each attribute and scaling them
def get_mean_list(filter_df):
    mean_list = []
    for col in col_list:
        col_max = abs(filter_df[col]).max()
        col_mean = abs(filter_df[col]).mean()
        col_value = col_mean/col_max
        mean_list.append(col_value)
    mean_list = [1-i for i in mean_list]
    return mean_list

### Functions for the Graphs ###

# Geting the info of the client
def get_index(client):
    client_index = list(client_df[client_df['SK_ID_CURR'] == client].index)[0]
    return client_index

# Getting the scaled values
def get_values(client_index, filter_df):
    client_values = []
    for col in col_list:
        value = abs(client_df.loc[client_index, col])
        col_max = abs(filter_df[col]).max()
        client_value = value/col_max
        client_values.append(client_value)
    client_values = [1-i for i in client_values]
    return client_values

# Get the title for graph
def get_title_spec(graph_title):
    title_specification = {
        'text':graph_title,
        'y':0.9,
        'x':0.5,
        'xanchor':'center',
        'yanchor':'top',
    }
    return title_specification

# Variables for the Radial Layout
radial_polar = {
            'radialaxis':{
                'visible':True,
                'range':[0,1]
            }
        }

radial_legend_spec = {
            'x':0,
            'y':1.1,
            'xanchor':'center',
            'yanchor':'top',
        }

#################################################################
### General Callbacks ###
    
# Checking the ID and returning the possibility of loan
@frontend.callback(
    Output('check_input', 'children'),
    Output('check_input', 'color'),
    Output('client_id', 'data'),
    Input('client_input', 'value'),
)
def check_input(input_value):
    if input_value in ['', None]:
        check_text = ''
        check_color = 'light'
        ID_temp = None
    else:
        try :
            ID_temp = int(input_value)
            if ID_temp in list(client_df['SK_ID_CURR']):
                try : 
                    storing_id(ID_temp)
                    check_text = "Ce client est bien dans la base de données."
                    check_color = 'success'
                except : 
                    check_text = 'Problem with storing'
                    check_color = 'warning'
            else:
                check_text = "L'identifiant entré n'est pas dans la base de données."
                check_color = 'info'
                ID_temp = None
        except:
            check_text = "Attention! L'identifiant ne doit être composé QUE de CHIFFRES."
            check_color = 'warning'
            ID_temp = None
    return check_text, check_color, ID_temp

# Getting the result of the loan and the probability
@frontend.callback(
    Output('result_text','children'),
    Output('proba_text','children'),
    Output('result_text','color'),
    Output('proba_text','color'),
    Output('client_score', 'data'),
    Output('client_proba', 'data'),
    Input('client_id', 'data')
)
def predict_loan(input_value):
    if input_value in ['', None]:
        result_text = "En attente d'itentifiant valide."
        proba_text = "En attente d'itentifiant valide."
        result_color = 'light'
        proba_color = 'light'
        client_score = None
        client_proba = None
    else:
        try:
            prediction = predict_score()
            client_score = prediction['client score']
            client_proba = prediction['client proba']
            if client_score == 0:
                result_text = f"Malheureusement, le client {input_value} n'est pas éligible à un prêt."
                result_color = 'danger'
                proba_text = f'La certitude est de {client_proba}%.'
                proba_color = 'danger'
            else:
                result_text = f"Le client {input_value} est éligible à un prêt."
                result_color = 'success'
                proba_text = f'La certitude est de {client_proba}%.'
                proba_color = 'success'
        except:
            result_text = 'Veuillez rentrer un identifiant valide.'
            proba_text = 'Veuillez rentrer un identifiant valide.'
            result_color = 'light'
            proba_color = 'light'
            client_score = None
            client_proba = None
    return result_text, proba_text, result_color, proba_color, client_score, client_proba

#################################################################
### Graphics Callbacks ###

# Graphique 1
@frontend.callback(
    Output('graph_a', 'figure'),
    Input('client_id', 'data')
)
def update_graph_a(client):
    # Keeping only the columns of interest
    filter_df = accepted_clients_df[col_list]
    mean_list = get_mean_list(filter_df)

    # Variables for the graph
    base_trace = go.Scatterpolar(
            r=mean_list,
            theta=graph_col,
            fill='toself',
            name='Moyenne des Clients Acceptés',
            line_color = 'green',
        )

    # If there is no client inputed
    if client == None:
        fig = go.Figure()
        fig.add_trace(base_trace)
        graph_title = 'Scores de Référence pour les Attributs Principaux'
        fig.update_layout(
            polar=radial_polar,
            showlegend=True,
            title=get_title_spec(graph_title),
            legend=radial_legend_spec
        )
    else:
        client_index = get_index(client)
        client_values = get_values(client_index, filter_df)

        # Displaying Graph
        fig = go.Figure()
        fig.add_trace(base_trace)
        fig.add_trace(go.Scatterpolar(
            r=client_values,
            theta=graph_col,
            fill='toself',
            name='Client',
            line_color='purple'
        ))
        graph_title = 'Comparaison entre les Scores de Référence et le Client'
        fig.update_layout(
            polar=radial_polar,
            showlegend=True,
            title=get_title_spec(graph_title),
            legend=radial_legend_spec
        )

    return fig

# Graphique 2
@frontend.callback(
    Output('graph_b', 'figure'),
    Output('collapse_graph_b', 'is_in'),
    Output('graph_b_title', 'children'),
    Input('graph_a', 'clickData'),
    Input('client_id', 'data')
)
def update_graph_b(attribute, client):
    try:
        gotten_col = attribute['points'][0]['theta']
        col_to_show = graph_dict[gotten_col]
        collapse_state = True
    except:
        col_to_show = 'EXT_SOURCE_MIN'
        gotten_col = 'Score Extérieur'
        collapse_state = False
    
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        x=refused_clients_df[col_to_show],
        name='Clients Refusés',
        marker_color='#FF4136',
        hoverinfo='skip'
    ))

    if client != None: 
        client_index=get_index(client)
        fig.add_trace(go.Box(
            x=[client_df.loc[client_index, col_to_show]],
            name='Client',
            marker_color = 'purple',
        ))
    
    fig.add_trace(go.Box(
        x=accepted_clients_df[col_to_show],
        name='Clients Acceptés',
        marker_color='#3D9970',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        xaxis={
            'title':f"{gotten_col}",
            'zeroline':False,
        },
        title={
        'text':f'Comparaison entre le Client et les Clients Acceptés et Refusés',
        'y':0.9,
        'x':0.5,
        'xanchor':'center',
        'yanchor':'top',
        }
    )
    fig.update_traces(orientation='h')
    graph_title_text = f"Informations complémentaire sur l'attribut {gotten_col}."

    return fig, collapse_state, graph_title_text


#################################################################
#################################################################
### Running the app ###

if __name__ == '__main__':
    frontend.run_server(debug=True)

#################################################################
#################################################################
