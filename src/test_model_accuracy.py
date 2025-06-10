import pandas as pd
import requests
import sys
from sklearn.metrics import accuracy_score
import json
import os

# Configuration
TEST_FEATURES_PATH = 'data/processed_test_data.csv' # Path for X_test
TEST_LABELS_PATH = 'data/y_test.csv'             # Path for y_test
API_URL = 'http://172.17.0.1:5001/predict' # URL of the staging model
ACCURACY_THRESHOLD = 0.90 # Define your accuracy threshold

print(f"--- Running Model Validation in Staging ---")
print(f"Loading test features from: {TEST_FEATURES_PATH}")
print(f"Loading test labels from: {TEST_LABELS_PATH}")

try:
    # Load the test data
    # Separate features (X) and true labels (y)
    # Ensure these column names match what your model was trained on and expects
	# Load the test features
    X_test = pd.read_csv(TEST_FEATURES_PATH)
    # Load the true labels for the test set
    y_true = pd.read_csv(TEST_LABELS_PATH).squeeze() # .squeeze() to convert DataFrame to Series

    print(f"Test data loaded. Number of samples: {len(X_test)}")

    predictions = []
    probabilities = []

    # Convert DataFrame to list of dictionaries for JSON payload
    payload = X_test.to_dict(orient='records')

    # Send prediction request to the API
    print(f"Sending {len(payload)} samples to {API_URL} for prediction...")
    headers = {'Content-Type': 'application/json'}
    response = requests.post(API_URL, data=json.dumps(payload), headers=headers)
    response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

    api_response = response.json()

    # Extract predictions and probabilities
    if not isinstance(api_response, list):
        raise ValueError(f"API response is not a list: {api_response}")

    for item in api_response:
        if 'prediction' not in item or 'probability' not in item:
            raise ValueError(f"API response item missing 'prediction' or 'probability': {item}")
        predictions.append(item['prediction'])
        probabilities.append(item['probability'])

    y_pred = pd.Series(predictions)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Model accuracy on staging environment: {accuracy:.4f}")

    # Check against threshold
    if accuracy >= ACCURACY_THRESHOLD:
        print(f"Accuracy {accuracy:.4f} meets or exceeds threshold {ACCURACY_THRESHOLD:.2f}. Model is APPROVED for production.")
        sys.exit(0) # Exit with success
    else:
        print(f"Accuracy {accuracy:.4f} is below threshold {ACCURACY_THRESHOLD:.2f}. Model is NOT APPROVED for production.")
        sys.exit(1) # Exit with failure

except FileNotFoundError:
    print(f"ERROR: Test data file not found at {TEST_DATA_PATH}. Ensure data_preprocessing.py ran correctly.")
    sys.exit(1)
except requests.exceptions.RequestException as e:
    print(f"ERROR: Could not connect to the model serving API at {API_URL}. Is the staging container running? Error: {e}")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"ERROR: Could not decode JSON response from API. Response content: {response.text}")
    sys.exit(1)
except ValueError as e:
    print(f"ERROR: Invalid API response format: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)
