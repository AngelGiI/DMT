import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Read the dataset
df = pd.read_csv('ODI-clean.csv', header=0, index_col=0, delimiter=';')

# Normalize continuous variables
scaler = MinMaxScaler()
df['How many hours per week do you do sports (in whole hours)? '] = scaler.fit_transform(df['How many hours per week do you do sports (in whole hours)? '].values.reshape(-1, 1))

# Convert categorical variables to numerical variables
le = LabelEncoder()
df['What programme are you in?'] = le.fit_transform(df['What programme are you in?'])
df['What is your gender?'] = le.fit_transform(df['What is your gender?'])
df['Have you taken a course on machine learning?'] = le.fit_transform(df['Have you taken a course on machine learning?'])
df['Have you taken a course on information retrieval?'] = le.fit_transform(df['Have you taken a course on information retrieval?'])
df['Have you taken a course on statistics?'] = le.fit_transform(df['Have you taken a course on statistics?'])
df['Have you taken a course on databases?'] = le.fit_transform(df['Have you taken a course on databases?'])
df['I have used ChatGPT to help me with some of my study assignments '] = le.fit_transform(df['I have used ChatGPT to help me with some of my study assignments '])
df['incorrect_answer'] = le.fit_transform(df['incorrect_answer'])
df['Good_weather_day'] = le.fit_transform(df['Good_weather_day'])
df['stress_level_bins'] = le.fit_transform(df['stress_level_bins'])

# Features and target
X = df[['What programme are you in?', 'Have you taken a course on databases?', 'What is your gender?', 'I have used ChatGPT to help me with some of my study assignments ','Good_weather_day']]
y = df['What is your stress level (0-100)?']

mse_scores = []
mae_scores = []
r2_scores = []

for i in range(10):
    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    
    # Create the decision tree regressor
    clf = DecisionTreeRegressor()
    
    # Fit the model
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate metrics
    mse_scores.append(mean_squared_error(y_test, y_pred))
    mae_scores.append(mean_absolute_error(y_test, y_pred))
    r2_scores.append(r2_score(y_test, y_pred))

# Write results to file
with open('ResultsTree.txt', 'w') as f:
    f.write("MSE for each run: {}\n".format(mse_scores))
    f.write("Average MSE: {:.4f}\n".format(np.mean(mse_scores)))
    f.write("MSE Standard Deviation: {:.4f}\n\n".format(np.std(mse_scores)))
    
    f.write("MAE for each run: {}\n".format(mae_scores))
    f.write("Average MAE: {:.4f}\n".format(np.mean(mae_scores)))
    f.write("MAE Standard Deviation: {:.4f}\n\n".format(np.std(mae_scores)))

    f.write("R-squared for each run: {}\n".format(r2_scores))
    f.write("Average R-squared: {:.4f}\n".format(np.mean(r2_scores)))
    f.write("R-squared Standard Deviation: {:.4f}\n".format(np.std(r2_scores)))