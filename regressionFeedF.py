import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
y = data['What is your stress level (0-100)?']

# Encode the categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the function to create the neural network model
def create_model(units_1=32, units_2=16, activation_1='relu', activation_2='tanh', optimizer_='sgd'):
    model = Sequential()
    model.add(Dense(units_1, input_dim=X_train.shape[1], activation=activation_1))
    model.add(Dense(units_2, activation=activation_2))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=optimizer_, metrics=['mse'])
    return model

# Create the KerasRegressor object
model = KerasRegressor(build_fn=create_model, epochs=50, batch_size=8)

# Define the hyperparameters to tune
param_grid = {
    'units_1': [16], #16,32,64
    'units_2': [8], #8,16,32
    'activation_1': ['tanh'], #relu, tanh
    'activation_2': ['tanh'], #relu, tanh
    'optimizer_': ['sgd'] #sgd, adam
}

# Perform grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=1)
grid_result = grid.fit(X_train, y_train)

# Print the best hyperparameters and the corresponding accuracy
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Predict on the test set and calculate the mean squared error and mean absolute error
y_pred = grid_result.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("MSE: %f" % mse)
print("MAE: %f" % mae)

# Best: -1111.561047 using {'activation_1': 'tanh', 'activation_2': 'tanh', 'optimizer_': 'sgd', 'units_1': 16, 'units_2': 8}
# MSE: 866.103329
# MAE: 25.060173

# Best: -1176.652930 using {'activation_1': 'tanh', 'activation_2': 'tanh', 'optimizer_': 'sgd', 'units_1': 16, 'units_2': 8}
# MSE: 1005.041387
# MAE: 25.012457