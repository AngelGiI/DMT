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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

# Remove the grid search and use the final tuned hyperparameters
units_1 = 50
units_2 = 25
activation_1 = 'relu'
activation_2 = 'tanh'
optimizer_ = 'sgd'

# Run the model for 10 iterations and save results to a file
n_runs = 10
results = []

for run in range(n_runs):
    model = create_model(units_1, units_2, activation_1, activation_2, optimizer_)
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)
    y_pred = model.predict(X_test)
    y_pred_class = np.argmax(y_pred, axis=1)
    y_test_class = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test_class, y_pred_class)
    precision = precision_score(y_test_class, y_pred_class, average='weighted')
    recall = recall_score(y_test_class, y_pred_class, average='weighted')
    f1 = f1_score(y_test_class, y_pred_class, average='weighted')

    results.append((accuracy, precision, recall, f1))

# Write results to file
with open('resultsFeedForward.txt', 'w') as f:
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