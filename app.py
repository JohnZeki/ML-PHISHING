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
import sys
import re

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


def extract_url_features(url):
    return {
        "url_length": len(url),
        "num_special_chars": len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', url)),
        "has_https": 1 if url.startswith("https://") else 0,
        "num_subdomains": url.count('.'),
        "contains_login": 1 if 'login' in url.lower() else 0,
        "contains_verify": 1 if 'verify' in url.lower() else 0,
    }


if __name__ == "__main__":
    # Check if a URL is provided via the command line
    if len(sys.argv) < 2:
        print("Usage: python app.py <url>")
        sys.exit(1)

    # Read the URL from the command line
    url = sys.argv[1]
    print(f"Analyzing URL: {url}")

    # Extract features from the URL
    features = extract_url_features(url)
    features_df = pd.DataFrame([features])

    # Load the trained model (if not already loaded in the script)
    with open('phishing_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    # Predict whether the URL is phishing or legitimate
    prediction = loaded_model.predict(features_df)[0]
    if prediction == 1:
        print("The URL is classified as: PHISHING")
    else:
        print("The URL is classified as: LEGITIMATE")
