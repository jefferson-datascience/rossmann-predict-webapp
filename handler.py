from flask import Flask, request, Response
from rossmann.Rossmann import Rossmann
import pandas as pd
import pickle
import os

# loading model
model = pickle.load(open('model/model_rossmann.pkl','rb'))

app = Flask(__name__)

@app.route('/rossmann/predict', methods=['POST'])

def rossman_predict():
    test_json = request.get_json()
    
    if test_json: # there is data
        
        if isinstance(test_json, dict): # Unique Example
            test_raw = pd.DataFrame(test_json, index=[0])
        
        else: # Multiple Example
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
        
        # Instaciando a classe rossman
        pipeline = Rossmann()
    
        # data_cleaning
        df1 = pipeline.data_cleaning(test_raw)
    
        # feature_engineering
        df2 = pipeline.feature_engineering(df1)

        # data_preparation
        df3 = pipeline.data_preparation(df2)

        # prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)

        return df_response
        
    else:
        return Response('{}', status=200, mimetype='application/json')
    

if __name__ == '__main__':
    port = os.environ.get('PORT', '5000')
    app.run(host='0.0.0.0', port=port)    