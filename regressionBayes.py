# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LinearRegression, BayesianRidge

# # Read the dataset
# df = pd.read_csv('ODI-2023_clean_mean.csv', header=0, index_col=0, delimiter=';')
# print(df.keys())

# # Normalize continuous variables
# scaler = MinMaxScaler()
# df['How many hours per week do you do sports (in whole hours)? '] = scaler.fit_transform(df['How many hours per week do you do sports (in whole hours)? '].values.reshape(-1, 1))

# # Convert categorical variables to numerical variables
# le = LabelEncoder()
# df['What programme are you in?'] = le.fit_transform(df['What programme are you in?'])
# df['What is your gender?'] = le.fit_transform(df['What is your gender?'])
# df['Have you taken a course on machine learning?'] = le.fit_transform(df['Have you taken a course on machine learning?'])
# df['Have you taken a course on information retrieval?'] = le.fit_transform(df['Have you taken a course on information retrieval?'])
# df['Have you taken a course on statistics?'] = le.fit_transform(df['Have you taken a course on statistics?'])
# df['Have you taken a course on databases?'] = le.fit_transform(df['Have you taken a course on databases?'])
# df['I have used ChatGPT to help me with some of my study assignments '] = le.fit_transform(df['I have used ChatGPT to help me with some of my study assignments '])
# df['incorrect_answer'] = le.fit_transform(df['incorrect_answer'])
# df['Good_weather_day'] = le.fit_transform(df['Good_weather_day'])
# df['What is your stress level (0-100)?'] = pd.to_numeric(df['What is your stress level (0-100)?'])

# # Features and target
# X = df[['What programme are you in?', 'Have you taken a course on databases?', 'What is your gender?', 'I have used ChatGPT to help me with some of my study assignments ', 'How many hours per week do you do sports (in whole hours)? ', 'Have you taken a course on information retrieval?', 'Have you taken a course on machine learning?', 'Have you taken a course on statistics?', 'incorrect_answer', 'Good_weather_day']]
# y = df['What is your stress level (0-100)?']

# # Create the Linear Regression model
# #clf = LinearRegression()
# clf = BayesianRidge()

# # Perform 5-fold cross-validation
# cv_scores = cross_val_score(clf, X, y, cv=5, scoring='neg_mean_squared_error')

# # Print the cross-validation scores and their average
# print("Cross-validation scores:", cv_scores)
# print("Average cross-validation score:", -np.mean(cv_scores), np.std(cv_scores))


###############################################################
###############################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Read the dataset
df = pd.read_csv('ODI-clean.csv', header=0, index_col=0, delimiter=';')
print(df.keys())

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

# Create the decision tree regressor
clf = DecisionTreeRegressor()

# Perform 5-fold cross-validation
cv_scores = cross_val_score(clf, X, y, cv=5, scoring='neg_mean_squared_error')
mse_scores = -cv_scores
print("MSE scores:", mse_scores)
print("Average MSE score:", np.mean(mse_scores))

cv_scores = cross_val_score(clf, X, y, cv=5, scoring='neg_mean_absolute_error')
mae_scores = -cv_scores
print("MAE scores:", mae_scores)
print("Average MAE score:", np.mean(mae_scores))
