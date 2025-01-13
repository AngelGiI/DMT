import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

# Read the dataset
df = pd.read_csv('ODI-clean.csv', header=0, index_col=0, delimiter=';')
print(df.keys())

# Example of transforming the stress levels to categories
def label_stress_level(level):
    if level < 30:
        return 'low' # low
    elif 30 <= level < 70:
        return 'medium' # medium
    else:
        return 'high' # high

df['What is your stress level (0-100)?'] = df['What is your stress level (0-100)?'].apply(label_stress_level)

# Convert categorical variables to numerical variables
le = LabelEncoder()
df['What programme are you in?'] = le.fit_transform(df['What programme are you in?'])
df['What is your gender?'] = le.fit_transform(df['What is your gender?'])

# Normalize continuous variables
scaler = MinMaxScaler()
df['How many hours per week do you do sports (in whole hours)? '] = scaler.fit_transform(df['How many hours per week do you do sports (in whole hours)? '].values.reshape(-1, 1))

df['Have you taken a course on machine learning?'] = le.fit_transform(df['Have you taken a course on machine learning?'])
df['Have you taken a course on information retrieval?'] = le.fit_transform(df['Have you taken a course on information retrieval?'])
df['Have you taken a course on statistics?'] = le.fit_transform(df['Have you taken a course on statistics?'])
df['Have you taken a course on databases?'] = le.fit_transform(df['Have you taken a course on databases?'])
df['I have used ChatGPT to help me with some of my study assignments '] = le.fit_transform(df['I have used ChatGPT to help me with some of my study assignments '])
df['What is your stress level (0-100)?'] = le.fit_transform(df['What is your stress level (0-100)?'])
# Features and target
X = df[['What is your stress level (0-100)?', 'Have you taken a course on machine learning?', 'Have you taken a course on information retrieval?', 'Have you taken a course on statistics?', 'Have you taken a course on databases?', 'What is your gender?', 'I have used ChatGPT to help me with some of my study assignments ']]
y = df['What programme are you in?']

# Create the Naive Bayes classifier
clf = GaussianNB()

# Perform 5-fold cross-validation
cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

# Print the cross-validation scores and their average
print("Cross-validation scores:", cv_scores)
print("Average cross-validation score:", np.mean(cv_scores))

