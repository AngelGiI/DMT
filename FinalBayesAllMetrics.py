import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer

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

# Create the Naive Bayes classifier
clf = GaussianNB()

# Perform 10 runs of 5-fold cross-validation
n_runs = 10
results = []

for run in range(n_runs):
    
    # Create a StratifiedKFold instance with 5 splits and a specified random state for reproducibility
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=run)

    # Calculate scores
    accuracy = cross_val_score(clf, X, y, cv=cv, scoring='accuracy').mean()
    precision = cross_val_score(clf, X, y, cv=cv, scoring=make_scorer(precision_score, average='weighted')).mean()
    recall = cross_val_score(clf, X, y, cv=cv, scoring=make_scorer(recall_score, average='weighted')).mean()
    f1 = cross_val_score(clf, X, y, cv=cv, scoring=make_scorer(f1_score, average='weighted')).mean()

    results.append((accuracy, precision, recall, f1))

# Write results to file
with open('resultsBayes.txt', 'w') as f:
    for i, result in enumerate(results):
        f.write(f"Run {i+1}:\n")
        f.write(f"Accuracy: {result[0]:.2f}\n")
        f.write(f"Precision: {result[1]:.2f}\n")
        f.write(f"Recall: {result[2]:.2f}\n")
        f.write(f"F1-score: {result[3]:.2f}\n\n")

    # Calculate average results
    avg_accuracy = np.mean([r[0] for r in results])
    avg_precision = np.mean([r[1] for r in results])
    avg_recall = np.mean([r[2] for r in results])
    avg_f1 = np.mean([r[3] for r in results])

    f.write("Average results:\n")
    f.write(f"Accuracy: {avg_accuracy:.2f}\n")
    f.write(f"Precision: {avg_precision:.2f}\n")
    f.write(f"Recall: {avg_recall:.2f}\n")
    f.write(f"F1-score: {avg_f1:.2f}\n")