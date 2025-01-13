import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load and preprocess the dataset
data = pd.read_csv('your_data.csv')
selected_features = [
    'What programme are you in?',
    'Have you taken a course on machine learning?',
    'Have you taken a course on information retrieval?',
    'Have you taken a course on statistics?',
    'Have you taken a course on databases?',
    'What is your gender?',
#    'I have used ChatGPT to help me with some of my study assignments',
#    'How many hours per week do you do sports (in whole hours)?'
]

X = data[selected_features]
y = data['What is your stress level (0-100)?']

# Transform the stress level into categorical values (low, medium, high)
def categorize_stress_level(value):
    if value < 30:
        return 'low'
    elif 30 <= value < 70:
        return 'medium'
    elif value == 'NaN':
        return 'medium'
    else:
        return 'high'

y = y.apply(categorize_stress_level)

# Encode the categorical variables
X = pd.get_dummies(X, drop_first=True)
y = LabelEncoder().fit_transform(y)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Create the neural network model
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 output classes: low, medium, high

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)
