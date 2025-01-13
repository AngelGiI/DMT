import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

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
le = LabelEncoder()
y = le.fit_transform(y)
n_classes = len(le.classes_)

def create_model(input_dim):
    model = Sequential()
    model.add(Dense(32, input_dim=input_dim, activation='relu'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

def errortrain_func(model, X_train, y_train, cv=5):
    scores = cross_val_score(model, X_train, y_train, cv=cv)
    return 1 - scores.mean()

def backward_selection_heuristic(X, y, create_model_func, errortrain_func):
    best_errortest = float('inf')
    best_selected_features = X.columns
    n = X.shape[0]
    
    for feature_to_remove in X.columns:
        # Remove the feature from the dataset
        X_temp = X.drop(feature_to_remove, axis=1)
        
        # Split the dataset into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_temp, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Create the KerasClassifier object
        model = KerasClassifier(build_fn=lambda: create_model_func(X_train.shape[1]), epochs=50, batch_size=32, verbose=0)
        
        # Perform 5-fold cross-validation
        errortrain = errortrain_func(model, X_train, y_train, cv=5)
        v = X_temp.shape[1]
        
        # Calculate errortest using the formula
        errortest = errortrain * (n + v) / (n - v)
        
        # Update the best model if the new errortest is lower
        if errortest < best_errortest:
            best_errortest = errortest
            best_selected_features = X_temp.columns
            
    return best_selected_features, best_errortest

# Use backward_selection_heuristic to find the best subset of features
best_features, best_errortest = backward_selection_heuristic(X, y, create_model, errortrain_func)
print("Best features:", best_features)
print("Best estimated test error: {:.4f}".format(best_errortest))
