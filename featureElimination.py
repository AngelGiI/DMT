import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

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
df['stress_level_bins'] = le.fit_transform(df['stress_level_bins'])
df['Good_weather_day'] = le.fit_transform(df['Good_weather_day'])

# Features and target
X = df[['What programme are you in?', 
        'Have you taken a course on machine learning?', 
        'Have you taken a course on information retrieval?', 
        'Have you taken a course on statistics?', 
        'Have you taken a course on databases?', 
        'What is your gender?', 
        'I have used ChatGPT to help me with some of my study assignments ',
        'Good_weather_day']]
y = df['stress_level_bins']

# Create the logistic regression classifier
clf = LogisticRegression(random_state=42, max_iter=1000)

# Perform RFE with 5 features
selector = RFE(clf, n_features_to_select=4)
selector.fit(X, y)

# Get the selected features
selected_features = X.columns[selector.support_]
print("Selected features:", selected_features)

# Create a new feature set with the selected features
X_selected = X[selected_features]

# Perform 5-fold cross-validation
cv_scores = cross_val_score(clf, X_selected, y, cv=10, scoring='accuracy')

# Print the cross-validation scores and their average
print("Cross-validation scores:", cv_scores)
print("Average cross-validation score:", np.mean(cv_scores))
