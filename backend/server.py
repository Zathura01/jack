from flask import Flask, jsonify, request
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
from catboost import Pool
import pandas as pd
import joblib
from utils import create_dataframe, fare_to_probability
from rose import preds
modelFare = joblib.load('C:/web projects/ALW70-ROSE/fare_Catpredictor_pipeline.pkl')




@app.route('/api/fare', methods=['POST'])
def fare():
    data = request.get_json()
    features = ['Embarked', 'FamilySize', 'Age', 'Sex', 'CabinPrefix_mean', 'Pclass', 'FamilyStatus']
    cat_features = ['Embarked', 'Sex', 'Pclass', 'FamilyStatus']  
    
    df = create_dataframe(data)

    df = df[features]
    pool = Pool(df, cat_features=cat_features)

    predictions = modelFare.predict(pool)
    probabs = [fare_to_probability(pred, data.get("Pclass", 3)) for pred in predictions]
    
    fare = 0
    for pr in predictions:
        fare = fare + pr
        
    fare = fare/len(predictions)

    return jsonify({
        'fare' : fare,
        'prediction': predictions.tolist(),
        'probability': probabs
    })



@app.route('/api/pred', methods=['POST'])
def pred():

    data = request.get_json()
    res = preds(data)

    return jsonify(res)


if __name__ == '__main__':
    app.run(debug=True, port=5000)