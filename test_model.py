import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import *


data = pd.read_csv('./data/census.csv')
data = data.drop_duplicates()

train, test = train_test_split(data, test_size=0.20)

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

X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features, 
    label='salary', 
    training=True, 
    encoder=None, 
    lb=None
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features, 
    label='salary', 
    training=False, 
    encoder=encoder, 
    lb=lb
)
# Building test functions
def test_train_model():
    '''
    
    Function to test the train_model function in model.py file
    
    '''
    model = train_model(X_train,y_train)
    assert model.get_params()['max_depth'] == 5
    assert type(model) == RandomForestClassifier

def test_inference():
    '''
    
    Function to test the inference function in model.py file

    '''
    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    assert len(preds) == len(X_train) #Assert that the length of the preds and X_train is same
    assert np.all((preds==0)|(preds == 1)) == True #To identify cases where the prediction values are not 0 and 1

def test_compute_model_metrics():
    '''
    
    Function to test the compute_model_metrics function in model.py file
    
    '''
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    metrics = compute_model_metrics(y_test, preds)
    assert len(metrics) == 3 #length of metrics
    assert type(metrics) == tuple #type of metrics
    for metric in metrics:
        assert metric >=0 and metric <= 1 #to ensure all metrics values are between 0 and 1

#if __name__ =="__main__":
test_train_model()
test_inference()
test_compute_model_metrics()