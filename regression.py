import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Assuming your dataset is already preprocessed and cleaned
dataset = pd.read_csv('your_preprocessed_dataset.csv')

# Select the 8 suggested attributes
feature_columns = [
    'What programme are you in?',
    'Have you taken a course on machine learning?',
    'Have you taken a course on information retrieval?',
    'Have you taken a course on statistics?',
    'Have you taken a course on databases?',
    'What is your gender?',
    'I have used ChatGPT to help me with some of my study assignments',
    'How many students do you estimate there are in the room?'
]

X = dataset[feature_columns]
y = dataset['What is your stress level (0-100)?']

# Convert categorical variables to dummy variables
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
tree_regressor = DecisionTreeRegressor(random_state=42)
tree_regressor.fit(X_train, y_train)

# Make predictions
y_pred = tree_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R2 Score: {r2:.2f}')
