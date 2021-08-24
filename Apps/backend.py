import joblib
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output
from flask import Flask, escape, request

from frontend import (client_id, create_dash_application, path_home,
                      path_infos, result_color, result_text, regex)

# 100001 not favorable and 100106 favorable

#################################################################
# Initialisation of the flask API
backend = Flask(__name__)

#################################################################
# Initialisation of usefull variables
client_df = pd.read_csv('Apps/Ressources/application_test.csv')
predict_df = pd.read_csv('Apps/Ressources/test_set.csv')

#################################################################
# Initialisation of the front end
frontend_app = create_dash_application(backend)

#################################################################
### Functions
# Predicting the score of a client
def predict_score(client_id):
    client_masque = predict_df[predict_df['SK_ID_CURR']==client_id]

    model = joblib.load('Apps/Ressources/p7_model')

    loan_result = model.predict(client_masque)
    return loan_result
    
# Checking the ID and returning the possibility of loan
@frontend_app.callback(
    Output('result_text', 'children'),
    Output('result_text', 'color'),
    Output('infos_path', 'href'),
    Output('home_path', 'href'),
    Input('client_input', 'value')
)
def say_loan(input_value):
    if input_value in ['', None]:
        result_text = ''
        result_color = 'light'
        path_infos = f"/infos"
        path_home = f"/"
    else:
        try :
            ID_temp = int(input_value)
            path_infos = f"/infos?{ID_temp}"
            path_home = f"/?{ID_temp}"
            if ID_temp in list(client_df['SK_ID_CURR']):
                result = predict_score(ID_temp)
                if result == 0:
                    result_text = f'The client with ID {input_value} is not favorable for a loan.'
                    result_color = 'danger'
                else:
                    result_text = f'The client with ID {input_value} is favorable for a loan.'
                    result_color = 'success'
            else:
                result_text = 'The ID you entered is not in the database.'
                result_color = 'info'
        except:
            result_text = 'The ID must be composed ONLY of NUMBERS.'
            result_color = 'warning'
            path_infos = f"/infos"
            path_home = f"/"
    return result_text, result_color, path_infos, path_home

# Getting the ID in the infos page
@frontend_app.callback(
    Output('shown_client_id', 'children'),
    Input('client_id', 'data')
)
def retrieve_client_id(input_data):
    if input_data == None :
        text_shown = ''
    else :
        text_shown = f'The ID of the client you entered was : {input_data}.'

    return text_shown#, input_data

# Keeping the ID when returning to the home page
@frontend_app.callback(
    Output('client_input', 'value'),
    Input('home_path', 'href')
)
def keeping_client_id(input_value):
    temp_list = regex.split(input_value)
    if len(temp_list) > 1 :
        ID_temp = temp_list[-1]
    else : 
        ID_temp = None
    return ID_temp

#################################################################
# Running the app
if __name__ == '__main__':
    backend.run(debug=True)

#################################################################
#################################################################
