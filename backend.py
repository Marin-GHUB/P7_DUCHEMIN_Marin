import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

#################################################################
# Initialisation of the flask API

backend = Flask(__name__)

#################################################################
# Initialisation of usefull variables

client_df = pd.read_csv('Apps/Ressources/application_test.csv')
predict_df = pd.read_csv('Apps/Ressources/test_set.csv')
model = joblib.load('Apps/Ressources/p7_model')
client = {
    "id" : None,
    "loan_proba" : None,
    "loan_score" : None
}

#################################################################
### Functions

# Welcome page
@backend.route('/')
def welcome_view():
    return "<h1>Welcome to the backend restAPI. It is running right now."

# Getting and sending the ID of a client
@backend.route('/id', methods=['GET', 'POST'])
def id_process():
    if (request.method == 'POST'):
        id = request.get_json()
        client['id'] = id['id']
        return jsonify('ID updated')
    elif (request.method == 'GET'):
        return jsonify(client['id'])

# Predicting the result of a client
@backend.route('/predict', methods=['GET'])
def predict_score_proba():
    ID = client['id']
    client_masque = predict_df[predict_df['SK_ID_CURR']==ID]
    client['loan_proba'] = int(max(model.predict_proba(client_masque)[0])*10000)/100
    client['loan_score'] = int(model.predict(client_masque))
    return jsonify({
        'client id' : client['id'], 
        'client proba' : client['loan_proba'], 
        'client score' : client['loan_score']
        })

# Sending the proba of a client
@backend.route('/proba', methods=['GET'])
def send_proba():
    return jsonify(client['loan_proba'])

# Sending the score of a client
@backend.route('/score', methods=['GET'])
def send_score():
    return jsonify(client['loan_score'])


### Due to limitation in memory on heroku, this step has been done localy instead of remotely ###
#
## Preprocessing the application test for the graphics
#client_df['RESULT'] = np.nan
#client_df['PROBA'] = np.nan
#client_df['LOAN'] = np.nan
#client_df['CREDIT_TO_ANNUITY_RATIO'] = np.nan
#client_df['EXT_SOURCE_MIN'] = np.nan
#
#for i, v in client_df.iterrows():
#    # Creating the credit to annuity ratio variable
#    client_credit = v['AMT_CREDIT']
#    client_annuity = v['AMT_ANNUITY']
#    v['CREDIT_TO_ANNUITY_RATIO'] = client_credit / client_annuity
#
#    # Creating the minimum source external score variable
#    client_source = []
#    client_source.append(v['EXT_SOURCE_1'])
#    client_source.append(v['EXT_SOURCE_2'])
#    client_source.append(v['EXT_SOURCE_3'])
#    v['EXT_SOURCE_MIN'] = min(client_source)
#
#    # Creating the result variables
#    temp_ID = v['SK_ID_CURR']
#    client_masque = predict_df[predict_df['SK_ID_CURR']==temp_ID]
#    v['PROBA'] = int(max(model.predict_proba(client_masque)[0])*10000)/100
#    v['RESULT'] = int(model.predict(client_masque))
#    if v['RESULT'] == 0:
#        v['LOAN'] = 'Refusé'
#    else :
#        v['LOAN'] = 'Accepté'
#
#
### Then the dataset has been saved and it is the one used in place of 'application_test.csv' on the github repository ###
#
#client_df.to_csv('Apps/Ressources/preprocessed_application_test.csv', index=False)



# Sending the dataframe
@backend.route('/dataframe', methods=['GET'])
def send_dataframe():
    return jsonify(client_df.to_json(orient='records'))

#################################################################
# Running the app
if __name__ == '__main__':
    backend.run(debug=True)

#################################################################
#################################################################
