import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report


# Load and preprocess the dataset
data = pd.read_csv('ODI-clean.csv', delimiter=';', header=0, index_col=0)
selected_features = [
        'What programme are you in?',
        'Have you taken a course on databases?', 
        'What is your gender?', 
        'I have used ChatGPT to help me with some of my study assignments ',
        'Good_weather_day'
]

X = data[selected_features]
y = data['stress_level_bins']


# Encode the categorical variables
X = pd.get_dummies(X, drop_first=True)
y = LabelEncoder().fit_transform(y)
print(X)
print(y)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the function to create the neural network model
def create_model(units_1=32, units_2=16, activation_1='relu', activation_2='tanh', optimizer_='sgd'):
    model = Sequential()
    model.add(Dense(units_1, input_dim=X_train.shape[1], activation=activation_1))
    model.add(Dense(units_2, activation=activation_2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_, metrics=['accuracy'])
    return model

# Create the KerasClassifier object
model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=8)

# Define the hyperparameters to tune
param_grid = {
    'units_1': [50],
    'units_2': [25],
    'activation_1': ['relu'],
    'activation_2': ['tanh'],
    'optimizer_': ['sgd']
}
#################### WINNERS: b_s = 8, u_1 = 32, u_2 = 16, a_1 = 'relu', a_2 = 'tanh', o = 'sgd' ####################


# Perform grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=1)
grid_result = grid.fit(X_train, y_train)

# Print the best hyperparameters and the corresponding accuracy
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Print the accuracy for each hyperparameter combination
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
