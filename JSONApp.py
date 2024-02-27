import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score
import requests



# User-defined function to add a row to the DataFrame
def add_row(dataframe, values):
    date_now = datetime.now()
    numeric_date_now = date_now.timestamp()
    new_row = {'Date':date_now,'ART': values['ART'], 'CPM': values['CPM'], 'EPM': values['EPM'], 'ExPM': values['ExPM'], 'HTTP': values['HTTP']}
    dataframe = dataframe.append(new_row, ignore_index=True)
    return dataframe

csv_path = 'C:/Users/Deekshita/Desktop/new.csv'
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    columns = ['ART', 'CPM', 'EPM', 'ExPM', 'HTTP', 'Date', 'NumericDate']
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
    df1=df[['Date','ART','CPM','EPM','ExPM','HTTP']]
    df1 = add_row(df1, json_obj)

    # Save the updated DataFrame to CSV
    df1.to_csv(csv_path, index=False)

    # Display updated data
    print("\nUpdated Data:")
    print(df1)
else:
    print("Data already exists in the CSV file. Not appending.")

df2=pd.read_csv(csv_path)
df2['NumericDate'] = pd.to_datetime(df1['Date']).apply(lambda x: x.timestamp())
df2.to_csv('C:/Users/Deekshita/Desktop/new_with_numeric_date.csv', index=False)

print("\nUpdated Data:")
print(df2)

data_for_isolation_forest=df2[['NumericDate','ART','CPM','EPM','ExPM','HTTP']]
print(data_for_isolation_forest)

isolation_forest = IsolationForest(contamination=0.05)
outliers = isolation_forest.fit_predict(data_for_isolation_forest)
data_for_isolation_forest['IsOutlier'] = outliers
accuracy_iso = accuracy_score(data_for_isolation_forest['IsOutlier'] == -1, outliers == -1)
num_correct_outliers = (outliers == -1).sum()
print(f'Accuracy (Isolation Forest): {accuracy_iso}')
print(f'Number of Correctly Identified Outliers: {num_correct_outliers}')

last_added_record = data_for_isolation_forest.iloc[-1]
is_last_record_outlier = last_added_record['IsOutlier'] == -1
print(f'Is the last added test record an outlier? {is_last_record_outlier}')


plt.plot(data_for_isolation_forest['NumericDate'], data_for_isolation_forest['ART'], color='black', label='art')
plt.plot(data_for_isolation_forest['NumericDate'], data_for_isolation_forest['CPM'],color='blue', label='cpm')
plt.scatter(data_for_isolation_forest['NumericDate'][data_for_isolation_forest['IsOutlier'] == -1], data_for_isolation_forest['ART'][data_for_isolation_forest['IsOutlier'] == -1], color='red', label='Outlier')
plt.xlabel('Numeric Date')
plt.ylabel('Avg Response Time')
plt.legend()
plt.title('Isolation Forest Outliers')
plt.show()


---------------------------------------------------
new.csv
Date,ART,CPM,EPM,ExPM,HTTP
2023-07-07 16:30:00.000000,130,4,1,3,2
2023-07-08 06:30:00.000000,124,5,1,4,0
2023-07-08 13:30:00.000000,161,7,2,1,18
2023-07-10 00:30:00.000000,148,3,0,5,9
2023-07-10 07:30:00.000000,373,0,1,8,1
2023-07-11 04:30:00.000000,195,0,0,10,1
2023-07-11 11:30:00.000000,199,1,0,6,1
2023-07-11 18:30:00.000000,310,0,1,0,8
2023-07-14 02:30:00.000000,161,13,2,62,1
2023-07-14 09:30:00.000000,216,4,0,5,6
2023-07-14 16:30:00.000000,172,3,2,12,5
2023-07-14 23:30:00.000000,160,3,1,3,2
2023-07-18 04:30:00.000000,174,3,2,4,0
2023-07-18 11:30:00.000000,150,0,0,12,45
2023-07-18 18:30:00.000000,151,7,0,3,1
2024-02-26 23:53:26.046300,34,23,1,3,23
2024-02-27 01:50:38.328282,13436,3,10,3,23370
2024-02-27 01:51:05.551014,13436,3,10,3,23370





app.route('/train', methods=['POST'])
def train_model():
    try:
        # Assuming X_train is your training data
        df2=pd.read_csv(csv)
        data_for_isolation_forest=df2[['NumericDate','ART','CPM','EPM','ExPM','HTTP']]
        print(data_for_isolation_forest)
        model_filename = 'isolation_forest_model.joblib'
        train_and_save_model(data_for_isolation_forest, model_filename)
        response = {'status': 'success', 'message': 'Model trained and saved successfully.'}
    except Exception as e:
        response = {'status': 'error', 'message': str(e)}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)