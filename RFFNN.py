import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

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

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Run the algorithm for 10 independent runs
n_runs = 10
mse_list, mae_list, r2_list = [], [], []

for run in range(n_runs):
    print(f"Run: {run + 1}")
    
    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Define the function to create the neural network model
    def create_model():
        model = Sequential()
        model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(16, activation='tanh'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mse'])
        return model

    # Create the KerasRegressor object
    model = KerasRegressor(build_fn=create_model, epochs=50, batch_size=8)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set and calculate the mean squared error, mean absolute error, and R-squared
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mse_list.append(mse)
    mae_list.append(mae)
    r2_list.append(r2)

    print(f"MSE: {mse}, MAE: {mae}, R-squared: {r2}")

# Calculate averages and standard deviations
mse_avg, mse_std = np.mean(mse_list), np.std(mse_list)
mae_avg, mae_std = np.mean(mae_list), np.std(mae_list)
r2_avg, r2_std = np.mean(r2_list), np.std(r2_list)

# Save the results to 'ResultsRFFNN.txt'
with open('ResultsRFFNN.txt', 'w') as f:
    f.write("Results for 10 independent runs of Feed Forward Neural Network Regressor:\n")
    f.write("-" * 50 + "\n")

    for run_number, (mse_value, mae_value, r2_value) in enumerate(zip(mse_list, mae_list, r2_list), start=1):
        f.write(f"Run {run_number}:\n")
        f.write(f"MSE: {mse_value:.4f}\n")
        f.write(f"MAE: {mae_value:.4f}\n")
        f.write(f"R-squared: {r2_value:.4f}\n")
        f.write("\n")

    f.write("Summary Statistics:\n")
    f.write("-" * 50 + "\n")
    f.write("Mean Squared Error (MSE):\n")
    f.write(f"Mean: {mse_avg:.4f}\n")
    f.write(f"Standard Deviation: {mse_std:.4f}\n")
    f.write("\n")
    f.write("Mean Absolute Error (MAE):\n")
    f.write(f"Mean: {mae_avg:.4f}\n")
    f.write(f"Standard Deviation: {mae_std:.4f}\n")
    f.write("\n")
    f.write("R-squared (Coefficient of Determination):\n")
    f.write(f"Mean: {r2_avg:.4f}\n")
    f.write(f"Standard Deviation: {r2_std:.4f}\n")