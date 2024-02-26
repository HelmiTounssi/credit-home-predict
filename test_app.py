import pandas as pd
import mlflow
from flask import Flask, request, jsonify
from sklearn.metrics import classification_report
import pickle
import numpy as np
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

TEST_DATA_PATH = "../data/final/processed_application_train.csv"
app = Flask('home-credit-prediction')



def load_test_data():
    new_data = pd.read_csv(TEST_DATA_PATH)
    return new_data.head(10000)


def prepare_data(data):
    # Clean up column names
    data.columns = data.columns.str.strip()
    for colonne in data.select_dtypes(include=['float64']).columns:
        data[colonne] = data[colonne].astype('float32', errors='ignore')
    for colonne in data.select_dtypes(include=['int64']).columns:
        data[colonne] = data[colonne].astype('int32', errors='ignore')    
    # Convertir la colonne REG_CITY_NOT_LIVE_CITY en int32
    data['REG_CITY_NOT_LIVE_CITY'] = data['REG_CITY_NOT_LIVE_CITY'].astype('int32')
    data['REG_REGION_NOT_WORK_REGION'] = data['REG_REGION_NOT_WORK_REGION'].astype('int32')   
    data['FLAG_DOCUMENT_8'] = data['FLAG_DOCUMENT_8'].astype('int32')
    # Select features and target
    X = data[data.columns.difference(['TARGET'])]
    Y = data['TARGET']
   
    return X,Y


def make_predictions_with_model_registry_model(model_name):
    # Load the data and prepare it
    data = load_test_data()
    X, Y = prepare_data(data)
    
    model_version = 1

   
    # Load the model and make predictions
    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
        data['predictions'] = model.predict(X)
        #data['prediction_probs'] = model.predict_proba(X)[:, 1]
        
    except:
        raise 

    return data


def make_predictions_with_model_pickle():
    # Load the data and prepare it
    data = load_test_data()
    X, Y = prepare_data(data)
       
    # Load the model and make predictions
    try:
        # 1. Charger le modèle à partir du fichier pickle
        with open('./model.pkl', 'rb') as f:
            model = pickle.load(f)
        data['predictions'] = model.predict(X)
        #data['prediction_probs'] = model.predict_proba(X)[:, 1]
        
    except:
        raise 

    return data

@app.route('/test', methods=['GET', 'POST'])
def predict_endpoint():
    
    '''predictions = make_predictions_with_model_registry_model(
        model_name="lightgbm"
    )'''
    predictions = make_predictions_with_model_pickle()

    cl_report = classification_report(predictions['TARGET'], 
                                      predictions['predictions'], 
                                      digits=3, 
                                      output_dict=True,
    )

    print(cl_report)
    return jsonify(cl_report)


if __name__ == "__main__":
    app.run(debug=True, port=9696)

