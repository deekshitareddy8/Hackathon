# app.py
from flask import Flask, request, jsonify
import pandas as pd
from IsolationForest import train_and_save_model

app = Flask(__name__)

csv_path_og = 'C:/Users/Deekshita/Desktop/sample1.csv'
csv_path = 'C:/Users/Deekshita/Desktop/new.csv'
csv_path_numeric = 'C:/Users/Deekshita/Desktop/new_with_numeric_date.csv'
model_filename = 'isolation_forest_model.joblib'

def load_data():
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print('Error: The CSV file is empty. Reading from the main CSV file.')
        df = pd.read_csv(csv_path_og)
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M')
        print(df)
        df.to_csv(csv_path, index=False)

    df1 = df[['Date', 'ART', 'CPM', 'EPM', 'ExPM', 'HTTP','CPU','MEM']]
    df2 = pd.read_csv(csv_path)
    df2['NumericDate'] = pd.to_datetime(df1['Date']).apply(lambda x: x.timestamp())
    df2.to_csv(csv_path_numeric, index=False)
    print('Updated new numeric file too.')

# Load data when the app starts
load_data()

@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Load the preprocessed data
        df2 = pd.read_csv(csv_path_numeric)
        data_for_isolation_forest = df2[['NumericDate', 'ART', 'CPM', 'EPM', 'ExPM', 'HTTP','CPU','MEM']]
        print(data_for_isolation_forest)

        # Train and save the model
        train_and_save_model(data_for_isolation_forest, model_filename)

        response = {'status': 'success', 'message': 'Model trained and saved successfully.'}
    except Exception as e:
        response = {'status': 'error', 'message': str(e)}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
