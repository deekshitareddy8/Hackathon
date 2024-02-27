import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score
df = pd.read_csv('C:/Users/Deekshita/Desktop/sample1.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M')
df['NumericDate'] = df['Date'].apply(lambda x: x.timestamp())
df = df.dropna()
X = df[['NumericDate']]
y = df[['ART','CPM','EPM','ExPM','HTTP','CPU','MEM']]  
linear_model = LinearRegression()
linear_model.fit(X, y)
plt.plot(df['NumericDate'], df['ART'],color='black', label='art')
plt.plot(df['NumericDate'], df['CPM'],color='blue', label='cpm')
plt.plot(df['NumericDate'], df['EPM'],color='yellow', label='epm')
plt.plot(df['NumericDate'], df['ExPM'],color='green', label='expm')
plt.plot(df['NumericDate'], df['HTTP'],color='red', label='http')
plt.plot(df['NumericDate'], df['CPU'],color='orange', label='cpu')
plt.plot(df['NumericDate'], df['MEM'],color='pink', label='mem')
plt.xlabel('Date')
plt.ylabel('Features')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
test_date = '8/19/2023 13:00'
test_numeric_date = datetime.strptime(test_date, '%m/%d/%Y %H:%M').timestamp()
predicted_response_time = linear_model.predict([[test_numeric_date]])
print(f'Linear Regression Predictions for the User Input: {predicted_response_time[0]}')
test_data = pd.DataFrame({'ART': [predicted_response_time[0][0]],'CPM': [predicted_response_time[0][1]],'EPM': [predicted_response_time[0][2]],'ExPM': [predicted_response_time[0][3]],'HTTP': [predicted_response_time[0][4]],'CPU': [predicted_response_time[0][5]],'MEM': [predicted_response_time[0][6]],'NumericDate': [test_numeric_date]})
print(test_data)
combined_df = pd.concat([df,test_data], axis=0)
df1 = combined_df[['NumericDate','ART','CPM','EPM','ExPM','HTTP','CPU','MEM']]
print(combined_df)
print(df1)
isolation_forest = IsolationForest(contamination=0.05)
outliers = isolation_forest.fit_predict(df1)
# Assuming outliers is a boolean array where True represents an outlier
df1['IsOutlier'] = outliers

accuracy_iso = accuracy_score(df1['IsOutlier'] == -1, outliers == -1)
num_correct_outliers = (outliers == -1).sum()
print(f'Accuracy (Isolation Forest): {accuracy_iso}')
print(f'Number of Correctly Identified Outliers: {num_correct_outliers}')

# Plot all features
plt.plot(df1['NumericDate'], df1['ART'], color='black', label='art')
plt.plot(df1['NumericDate'], df1['CPM'], color='blue', label='cpm')
plt.plot(df1['NumericDate'], df1['EPM'], color='yellow', label='epm')
plt.plot(df1['NumericDate'], df1['ExPM'], color='green', label='expm')
plt.plot(df1['NumericDate'], df1['HTTP'], color='red', label='http')
plt.plot(df1['NumericDate'], df1['CPU'], color='orange', label='cpu')
plt.plot(df1['NumericDate'], df1['MEM'], color='pink', label='mem')

# Scatter plot for outliers with all features
outlier_points = df1[df1['IsOutlier'] == -1]
plt.scatter(outlier_points['NumericDate'], outlier_points['ART'], color='red')
plt.scatter(outlier_points['NumericDate'], outlier_points['CPM'], color='red')
plt.scatter(outlier_points['NumericDate'], outlier_points['EPM'], color='red')
plt.scatter(outlier_points['NumericDate'], outlier_points['ExPM'], color='red')
plt.scatter(outlier_points['NumericDate'], outlier_points['HTTP'], color='red')
plt.scatter(outlier_points['NumericDate'], outlier_points['CPU'], color='red')
plt.scatter(outlier_points['NumericDate'], outlier_points['MEM'], color='red')

plt.xlabel('Numeric Date')
plt.ylabel('Features')
plt.legend()
plt.title('Isolation Forest Outliers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


