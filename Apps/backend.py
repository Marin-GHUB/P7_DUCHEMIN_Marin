import pandas as pd
import joblib
import numpy as np


df = pd.read_csv('Apps/Ressources/test_set.csv')

row_number = np.random.randint(1, len(df))
client_id = df.loc[row_number, 'SK_ID_CURR']
client_masque = df[df['SK_ID_CURR']==client_id]

model = joblib.load('Apps/Ressources/p7_model')

loan_result = model.predict(client_masque)

if loan_result == 0:
    print("Le client {} ne peut pas avoir de pret.".format(client_id))
else:
    print('Le client {} peut avoir un pret.'.format(client_id))