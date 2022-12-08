import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  OneHotEncoder, LabelEncoder
import numpy as np


# Add code to load in the data, model and encoder
data = pd.read_csv('./data/census.csv')

data.columns = data.columns.str.strip()
data = data.drop_duplicates()
model = pd.read_pickle(r"model.pkl")
encoder = pd.read_pickle(r"encoder.pkl") 


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

_ , test_set = train_test_split(
    data, 
    test_size=0.20, 
    random_state=42, 
    stratify=data.salary
    )

lb = LabelEncoder() 

test_set['salary'] = lb.fit_transform(test_set['salary'])

X_test, y_test, _, _ = process_data(
                test_set.drop('salary',axis=1),
                cat_features,
                label= None, encoder=encoder, lb=lb, training=False)

y_preds=inference(model, X_test)

y =test_set.iloc[:,-1:]

y = lb.fit_transform(np.ravel(y))

prc, rcl, fb = compute_model_metrics(y, y_preds)
print(f"Precision: {prc} \nRecall: {rcl}\nFbeta Score: {fb}")

## Results:
## Precision: 0.8455696202531645 
## Recall: 0.4260204081632653
## Fbeta Score: 0.5665818490245971