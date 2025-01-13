import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

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
X = df[['What programme are you in?', 
        'Have you taken a course on machine learning?', 
        'Have you taken a course on information retrieval?', 
        'Have you taken a course on statistics?', 
        'Have you taken a course on databases?', 
        'What is your gender?', 
        'I have used ChatGPT to help me with some of my study assignments ',
        'incorrect_answer',
        'Good_weather_day']]
y = df['stress_level_bins']

# Backward selection
best_test_error = float('inf')
best_features = X.columns.tolist()
cv = 5  # Number of cross-validation folds

test_errors = []  # Store test errors and feature combinations

while len(best_features) > 1:
    current_features = best_features.copy()
    feature_removal = None
    improvement = False

    for feature in current_features:
        reduced_features = best_features.copy()
        reduced_features.remove(feature)
        reduced_X = X[reduced_features]

        # Perform k-fold cross-validation
        clf = GaussianNB()
        cv_scores = cross_val_score(clf, reduced_X, y, cv=cv, scoring='accuracy')

        # Calculate training error
        train_error = 1 - np.mean(cv_scores)

        # Calculate estimated test error
        n, v = len(reduced_X), len(reduced_features)
        test_error = train_error * (n + v) / (n - v)

        if test_error < best_test_error:
            improvement = True
            best_test_error = test_error
            feature_removal = feature

    # If test error increased, reset the best test error
    if not improvement:
        best_test_error = float('inf')
        continue

    # Save the test error and feature combination
    test_errors.append((best_test_error, best_features.copy()))

    best_features.remove(feature_removal)

print("Selected features:", best_features)

# Write test errors and feature combinations to a file
with open('test_errors2.txt', 'w') as f:
    for error, features in test_errors:
        f.write(f'Test error: {error:.4f}, Features: {", ".join(features)}\n')
