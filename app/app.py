import joblib
from flask import Flask, request, jsonify
import pandas as pd
import os
from flask_cors import CORS # <--- Add this line


app = Flask(__name__)
CORS(app) # <--- Add this line right after app initialization to enable CORS for all routes
# --- Model Loading (Global Scope) ---
# This will load the model once when the Flask app is initialized by Gunicorn.
model = None
model_path = os.environ.get('MODEL_PATH', 'fraud_model.pkl')
print(f"Flask app: Loading model from: {model_path}")
try:
    model = joblib.load(model_path)
    print("Flask app: Model loaded successfully!")
except Exception as e:
    print(f"Flask app: Error loading model: {e}")
    model = None
# --- End Model Loading ---


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json(force=True) # Get JSON data from the request body
    try:
        # Create a DataFrame from the input data.
        # Ensure the column names match the features used for training:
        # 'amount', 'amount_per_location', 'location', 'is_amex'
        input_df = pd.DataFrame(data)

        # Make predictions and get probabilities
        predictions = model.predict(input_df[['amount', 'amount_per_location', 'location', 'is_amex']])
        probabilities = model.predict_proba(input_df[['amount', 'amount_per_location', 'location', 'is_amex']])[:, 1]

        results = []
        for i in range(len(predictions)):
            results.append({
                'prediction': int(predictions[i]),
                'probability': float(probabilities[i])
            })
        return jsonify(results)
    except KeyError as e:
        return jsonify({'error': f"Missing expected feature: {e}. Required features: amount, amount_per_location, location, is_amex"}), 400
    except Exception as e:
        return jsonify({'error': f"Prediction failed: {e}"}), 500

@app.route('/health', methods=['GET'])
def health():
    if model is not None:
        return jsonify({'status': 'ok', 'model_loaded': True})
    else:
        return jsonify({'status': 'error', 'model_loaded': False}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
