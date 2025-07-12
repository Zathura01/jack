import pandas as pd
import numpy as np



cabin_mean_map = {
    None: 12.823204,
    'C': 58.883282,
    'E': 30.373543,
    'G': 5.083750,
    'D': 32.703774,
    'A': 31.608805,
    'CCC': 43.833333,
    'B': 65.891841,
    'F': 10.232143,
    'FG': 7.650000,
    'DD': 31.679150,
    'BB': 66.153467,
    'FE': 7.303722,
    'CC': 75.106775,
    'BBBB': 52.475000,
    'T': 35.500000,
    'BBB': 129.332300,
    'EE': 134.500000
}


def create_dataframe(data):

    df = []
    famSize = len(data.get("Family",[])) + 1
    famStat = famSize > 1
    cabinMean = cabin_mean_map.get(data.get("Cabin","S"))

    df.append({
        "Embarked": data.get("Embarked", "Q"),
        "FamilySize": famSize,
        "Age": data.get("Age", 0.0),
        "Sex": data.get("Sex", "male"),
        "CabinPrefix_mean": cabinMean,
        "Pclass": data.get("Pclass", 1),
        "FamilyStatus": famStat
    })

    # Add family member entries
    for fam in data.get("Family", []):
        df.append({
            "Embarked": fam.get("Embarked", data.get("Embarked", "Q")),
            "FamilySize": famSize,
            "Age": fam.get("Age", 0.0),
            "Sex": fam.get("Sex", "male"),
            "CabinPrefix_mean": cabinMean,
            "Pclass": fam.get("Pclass", data.get("Pclass", 1)),
            "FamilyStatus": famStat
        })


    return pd.DataFrame(df)
 
thresholds = {
    1:75.0,
    2:25.0,
    3:10.0
}

def fare_to_probability(fare, Pclass, sharpness=1.0):
    # Higher sharpness = steeper sigmoid
    prob_high = 1 / (1 + np.exp(-sharpness * (fare - thresholds.get(Pclass))))
    return [1 - prob_high, prob_high] 