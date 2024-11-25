# ================================
# Imports and Data Preparation
# ================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle  # To save the model
from flask import Flask, request, jsonify  # Flask for API deployment

# ================================
# Data Collection and Sanitation
# ================================

# Load the dataset
df = pd.read_csv('dataset_full.csv')

# Separate features and labels
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]  # The last column as the target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================
# Model Training
# ================================

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print metrics
print("Presnosť:", accuracy)
print("Precíznosť:", precision)
print("Citlivosť:", recall)
print("F1 skóre:", f1)
print("Matrica zámien:\n", confusion_matrix(y_test, y_pred))

# Save the trained model
with open('phishing_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# ================================
# Flask API for Deployment
# ================================

# Load the trained model
with open('phishing_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Create Flask application
app = Flask(__name__)


# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = pd.DataFrame([data])  # Expecting data as a dictionary of feature values
    prediction = loaded_model.predict(features)[0]
    return jsonify({"is_phishing": bool(prediction)})


if __name__ == '__main__':
    app.run(debug=True)
