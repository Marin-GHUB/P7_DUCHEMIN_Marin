from dash.dependencies import Input, Output
import pandas as pd
import joblib
import numpy as np
from flask import Flask, escape, request
from frontend import create_dash_application, result_text

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
    Output(component_id='result_text', component_property='children'),
    Input(component_id='client_id', component_property='value')
)
def say_ID(input_value):
    if input_value in ['', None]:
        result_text = ''
    else:
        try :
            int(input_value)
            if int(input_value) in list(client_df['SK_ID_CURR']):
                result = predict_score(int(input_value))
                if result == 0:
                    result_text = f'The client with ID {input_value} is not favorable for a loan.'
                else:
                    result_text = f'The client with ID {input_value} is favorable for a loan.'
            else:
                result_text = 'The ID you entered is not in the database.'
        except:
            result_text = 'The ID must be composed ONLY of NUMBERS.'
    #result_text = '{}'.format(i)
    return result_text

#################################################################
# Running the app
if __name__ == '__main__':
    backend.run(debug=True)

#################################################################
#################################################################