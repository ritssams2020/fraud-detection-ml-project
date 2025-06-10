import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
import os
import time

def evaluate_model(model_path="model/fraud_model.pkl", test_data_path="data/test_data.csv", metrics_output_path="metrics/evaluation_metrics.json"):
    print("--- Model Evaluation ---")
    if not os.path.exists('metrics'):
        os.makedirs('metrics') # Create the 'metrics' directory if it doesn't exist

    model = joblib.load(model_path)
    test_df = pd.read_csv(test_data_path)

    X_test = test_df[['amount', 'amount_per_location', 'location', 'is_amex']]
    y_test = test_df['is_fraud']

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probability of fraud

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f, indent=4) # Save metrics as a JSON file

    print("\n--- Evaluation Results ---")
    for metric, value in metrics.items():
        print(f"   {metric.replace('_', ' ').capitalize()}: {value:.4f}")
    print("--------------------------\n")
    print(f"Evaluation metrics saved to {metrics_output_path}")
    time.sleep(1) # Simulate work

    # Simple quality gate: if F1-score is too low, the script will indicate failure.
    # In a real Jenkins job, this will cause the build step to fail.
    if f1 < 0.7:
        print("ERROR: F1-Score is below threshold (0.7)! Model might not be good enough.")
        return False # Indicate failure
    return True # Indicate success

if __name__ == "__main__":
    if not evaluate_model():
        exit(1) # Exit with a non-zero code to signal failure
