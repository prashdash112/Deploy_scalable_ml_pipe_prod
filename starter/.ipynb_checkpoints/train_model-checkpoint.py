# Script to train machine learning model.
# Add the necessary imports for the starter code.
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model 

# Add code to load in the data.
path = '../'
data = pd.read_csv('../data/census.csv')
df = df.drop_duplicates()

# Optional enhancement, use K-fold cross validation instead of a train-test split.
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

# Proces the train data with the process_data function.
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

# Train and save a model.
model = train_model(X_train,y_train)
pd.to_pickle(model, os.path.join(path, "model.pkl"))
pd.to_pickle(model, os.path.join(path, "encoder.pkl"))
pd.to_pickle(model, os.path.join(path, "lb.pkl"))