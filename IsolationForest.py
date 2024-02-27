# train_model.py
import pandas as pd
from sklearn.ensemble import IsolationForest
from joblib import dump

def train_and_save_model(X_train, model_filename):
    # Train the model
    isolation_forest = IsolationForest(contamination=0.05)
    isolation_forest.fit(X_train)

    # Save the trained model
    
    dump(isolation_forest, model_filename)

    print("Model trained and saved successfully.")


