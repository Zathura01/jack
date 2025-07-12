from flask import Flask, request, jsonify

import joblib 
import pandas as pd
import numpy as np
from flask import jsonify


model_one = joblib.load('C:/web projects/ALW70-ROSE/titanic_model_One.pkl')
model_two = joblib.load('C:/web projects/ALW70-ROSE/titanic_model_Two.pkl')
model_three = joblib.load('C:/web projects/ALW70-ROSE/titanic_model_Three.pkl')
preprocessor = joblib.load('C:/web projects/ALW70-ROSE/titanic_preprocessor_Three.pkl')


def assignwealthcombo(wealthstat, fare, pclass):
    if ( fare > 74 and pclass == 2) or ( fare > 25 and pclass == 1):
            return wealthstat.append(1) 
    elif ( fare < 25 and ( pclass == 1 or pclass == 2)) or ( fare > 10 and ( pclass == 2 or pclass == 3)):
            return wealthstat.append(2)
    elif fare < 10 and ( pclass == 3 or pclass == 2):
            return wealthstat.append(3)
    else:
            return wealthstat.append(0)


def assign_wealth_status(wealthStat, data):
    
    assignwealthcombo(wealthStat, data.get("Fare", 15), data.get("Pclass", 3))

    for d in data.get("Family",[]): 
        fare = d.get("Fare", 15)
        pclass = d.get("Pclass", 2)
        assignwealthcombo(wealthStat, fare, pclass)

    return wealthStat        


def modelonedf (data):

    wealthStat = []
    wealthStat = assign_wealth_status(wealthStat, data)
    df = []
    famSize = len(data.get("Family", [])) + 1
    famStat = famSize > 1

    df.append(
        {
            "Pclass": data.get("Pclass", 3),
            "Sex": data.get("Sex", 'male'),
            "Age": data.get("Age", 18),
            "SibSp": data.get("SibSp", 0),
            "Parch": data.get("Parch", 0),
            "Fare": data.get("Fare", 10),
            "Embarked": data.get("Embarked", 'C'),
            "FamilySize": famSize,
            "FamilyExist": famStat,
            "WealthStatus": wealthStat[0]
        }
    )
    i = 0
    for fam in data.get("Family", []):
          i = i + 1
          df.append({
            "Pclass": data.get("Pclass", 3),
            "Sex": fam.get("Sex", 'male'),
            "Age": fam.get("Age", 18),
            "SibSp": fam.get("SibSp", 0),
            "Parch": fam.get("Parch", 0),
            "Fare": fam.get("Fare", 10),
            "Embarked": fam.get("Embarked", 'C'),
            "FamilySize": famSize,
            "FamilyExist": famStat,
            "WealthStatus": wealthStat[i]
          })

    return pd.DataFrame(df)


def modeltwodf (data):
    df = []

    df.append(
        {
            "Pclass": data.get("Pclass", 3),
            "Age": data.get("Age", 18),
            "Sex": data.get("Sex", 'male'),
            "Embarked": data.get("Embarked", 'C'),
            "Fare": data.get("Fare", 10)
        }
    )

    for fam in data.get("Family", []):
          df.append({
            "Pclass": data.get("Pclass", 3),
            "Age": fam.get("Age", 18),
            "Sex": fam.get("Sex", 'male'),
            "Embarked": fam.get("Embarked", 'C'),
            "Fare": fam.get("Fare", 10)
          })

    return pd.DataFrame(df)



def modelthreedf (data):

    df = []
    df.append(
        {
            "Age": data.get("Age", 18),
            "Sex": data.get("Sex", 'male'),
            "Pclass": data.get("Pclass", 3),
            "Fare": data.get("Fare", 10),
            "Embarked": data.get("Embarked", 'C')
        }
    )

    for fam in data.get("Family", []):
          df.append({
            "Age": fam.get("Age", 18),
            "Sex": fam.get("Sex", 'male'),
            "Pclass": data.get("Pclass", 3),
             "Fare": fam.get("Fare", 10),
            "Embarked": fam.get("Embarked", 'C')
          })

    return pd.DataFrame(df)


def preds(data):
    all_predictions = {}

    # Model One
    df1 = modelonedf(data)
    prediction = model_one.predict(df1)
    probability = model_one.predict_proba(df1)
    result_one = []
    for i in range(len(prediction)):
        result_one.append({
            'prediction': int(prediction[i]),
            'probability': {
                'Not Survived': round(probability[i][0], 4),
                'Survived': round(probability[i][1], 4)
            }
        })
    all_predictions['model_one'] = result_one

    # Model Two
    df2 = modeltwodf(data)
    preds2 = model_two.predict(df2)
    probs2 = model_two.predict_proba(df2)
    result_two = []
    for i in range(len(preds2)):
        result_two.append({
            'prediction': int(preds2[i]),
            'probability': {
                'Not Survived': round(probs2[i][0], 4),
                'Survived': round(probs2[i][1], 4)
            }
        })
    all_predictions['model_two'] = result_two

    # Model Three
    df3 = modelthreedf(data)
    X = preprocessor.transform(df3)
    preds3 = model_three.predict(X)
    probs3 = model_three.predict_proba(X)
    result_three = []
    for i in range(len(preds3)):
        result_three.append({
            'prediction': int(preds3[i]),
            'probability': {
                'Not Survived': round(probs3[i][0], 4),
                'Survived': round(probs3[i][1], 4)
            }
        })
    all_predictions['model_three'] = result_three
    print(all_predictions)
    
    return all_predictions

# ONE MODEL
# Pclass	Sex	Age	SibSp	Parch	Fare	Embarked	FamilySize	FamilyExist	WealthStatus
# 709	3	male	NaN	1	1	15.2458	C	3	True	2
# 439	2	male	31.0	0	0	10.5000	S	1	False	2

# TWO MODEL
# 	Pclass	Age	Sex	Embarked	Fare
# 0	3	22.0	male	S	7.2500
# 1	1	38.0	female	C	71.2833

# THREE MODEL
# colm = ['Age','Sex','Pclass','Fare','Embarked']
