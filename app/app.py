import joblib
from flask import Flask, request, jsonify
import pandas as pd
import os

app = Flask(__name__)
model = None

# Load model on startup
@app.before_first_request
def load_model():
    global model
    # Model path can be set via environment variable for flexibility in Docker
    model_path = os.environ.get('MODEL_PATH', 'fraud_model.pkl')
    print(f"Loading model from: {model_path}")
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

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
    # For production, you'd use a WSGI server like Gunicorn, which our Dockerfile uses.
    app.run(host='0.0.0.0', port=5000)
