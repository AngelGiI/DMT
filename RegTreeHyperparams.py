import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
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

# Specify hyperparameters and their distributions
param_dist = {
    'max_depth': np.arange(3, 11),
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [5, 10, 20],
    'max_features': ['sqrt', 'log2', 0.5],
    'max_leaf_nodes': [10, 20, 30],
    'ccp_alpha': np.linspace(0, 0.1, 11)
}

# Perform randomized search for hyperparameters
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=50, cv=5, scoring='neg_mean_squared_error', random_state=42)
random_search.fit(X, y)

# Get the best hyperparameters
best_params = random_search.best_params_
print("Best hyperparameters:", best_params)

# Train the model with the best hyperparameters
best_clf = DecisionTreeRegressor(**best_params)
cv_scores = cross_val_score(best_clf, X, y, cv=5, scoring='neg_mean_squared_error')
mse_scores = -cv_scores
print("MSE scores with best hyperparameters:", mse_scores)
print("Average MSE score with best hyperparameters:", np.mean(mse_scores))

cv_scores = cross_val_score(best_clf, X, y, cv=5, scoring='neg_mean_absolute_error')
mae_scores = -cv_scores
print("MAE scores with best hyperparameters:", mae_scores)
print("Average MAE score with best hyperparameters:", np.mean(mae_scores))
