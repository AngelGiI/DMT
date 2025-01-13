import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

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
y = df['stress_level_bins']

# Define the parameter grid for 'var_smoothing'
param_grid = {'var_smoothing': np.logspace(-15, -3, num=50)}

# Create the Naive Bayes classifier
clf = GaussianNB()

# Set up the grid search
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X, y)

# Print the best 'var_smoothing' value and the corresponding score
print("Best 'var_smoothing' value:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# You can also access the best estimator directly
best_clf = grid_search.best_estimator_