import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm


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

# Backward elimination with p-value threshold
p_value_threshold = 0.1
remaining_features = X.columns.tolist()

while True:
    X_sm = sm.add_constant(X[remaining_features])
    model = sm.MNLogit(y, X_sm).fit(disp=False)  # Fit logistic regression model
    p_values = model.pvalues
    max_p_value = p_values.max()

    if (p_values > p_value_threshold).any(axis=None):
        worst_feature_index = p_values.stack().argmax() - 1  # Account for the constant column
        worst_feature = remaining_features[worst_feature_index]
        print(f"Removing feature: {worst_feature}")
        remaining_features.remove(worst_feature)
    else:
        break

print("Selected features:", remaining_features)
