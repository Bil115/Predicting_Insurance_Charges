from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("xgb_final.pkl")

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Insurance Charges Prediction API!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        
        # Extract input features
        features = np.array(data["features"]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        
        # Return result
        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
