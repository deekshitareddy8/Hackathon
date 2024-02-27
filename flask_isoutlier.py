# flask_app.py
from flask import Flask, jsonify
import pandas as pd
from sklearn.ensemble import IsolationForest
from joblib import load
from datetime import datetime
from sklearn.metrics import accuracy_score
import requests
from IsolationForest import train_and_save_model  # Import the training function

app = Flask(__name__)

# User-defined function to add a row to the DataFrame fl
def add_row(dataframe, values):
    date_now = datetime.now()
    numeric_date_now = date_now.timestamp()
    new_row = {'Date': date_now, 'ART': values['ART'], 'CPM': values['CPM'], 'EPM': values['EPM'], 'ExPM': values['ExPM'], 'HTTP': values['HTTP'],'CPU': values['CPU'],'MEM': values['MEM']}
    dataframe = dataframe.append(new_row, ignore_index=True)
    return dataframe

csv_path = 'C:/Users/Deekshita/Desktop/new.csv'
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    columns = ['ART', 'CPM', 'EPM', 'ExPM', 'HTTP','CPU','MEM', 'Date', 'NumericDate']
    df = pd.DataFrame(columns=columns)

# Flask API URL
flask_api_url = 'http://127.0.0.1:5000/get_hardcoded_object'

# Make a request to the Flask API to get the JSON object
response = requests.get(flask_api_url)
json_obj = response.json()
print(json_obj)

# Check if the new data already exists based on the 'Date' column
formatted_date_now = datetime.now().strftime('%#m/%#d/%Y %H:%M')
if formatted_date_now not in df['Date'].values:
    # Add the test data to the DataFrame
    df1 = df[['Date', 'ART', 'CPM', 'EPM', 'ExPM', 'HTTP','CPU','MEM']]
    df1 = add_row(df1, json_obj)

    # Save the updated DataFrame to CSV
    df1.to_csv(csv_path, index=False)

    # Display updated data
    print("\nUpdated Data:")
    print(df1)
else:
    print("Data already exists in the CSV file. Not appending.")

df2 = pd.read_csv(csv_path)
df2['NumericDate'] = pd.to_datetime(df1['Date']).apply(lambda x: x.timestamp())
df2.to_csv('C:/Users/Deekshita/Desktop/new_with_numeric_date.csv', index=False)

print("\nUpdated Data:")
print(df2)

data_for_isolation_forest = df2[['NumericDate', 'ART', 'CPM', 'EPM', 'ExPM', 'HTTP','CPU','MEM']]
print(data_for_isolation_forest)

# Load the trained Isolation Forest model
model_filename = 'isolation_forest_model.joblib'
try:
    isolation_forest = load(model_filename)
except FileNotFoundError:
    isolation_forest = None
    print("Model not found. Please train the model using train_model.py.")

@app.route('/isoutlier')
def is_outlier():
    if isolation_forest:
        outliers = isolation_forest.fit_predict(data_for_isolation_forest)
        data_for_isolation_forest['IsOutlier'] = outliers
        last_added_record = data_for_isolation_forest.iloc[-1]
        is_last_record_outlier = last_added_record['IsOutlier'] == -1
        return jsonify({"isOutlier": bool(is_last_record_outlier)})
    else:
        return jsonify({"error": "Model not found. Please train the model using train_model.py."})

if __name__ == '__main__':
    app.run(debug=True,port=8000)
