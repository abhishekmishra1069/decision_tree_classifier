from flask import Flask, request, jsonify
import joblib
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('decision_tree_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return "Decision Tree Classification API"

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.get_json(force=True)
    age = data['age']
    salary = data['salary']

    # Scale the input features
    input_features = scaler.transform([[age, salary]])

    # Make prediction
    prediction = model.predict(input_features)

    # Return the prediction
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)