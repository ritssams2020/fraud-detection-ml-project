import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib # For saving/loading models
import os
import time

def train_model(features_path="data/features.csv", model_output_path="model/fraud_model.pkl"):
    print("--- Model Training ---")
    if not os.path.exists('model'):
        os.makedirs('model') # Create the 'model' directory if it doesn't exist

    df = pd.read_csv(features_path)
    X = df[['amount', 'amount_per_location', 'location', 'is_amex']]
    y = df['is_fraud']

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Save test data for evaluation stage
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv("data/test_data.csv", index=False)
    print("Test data saved for evaluation.")

    print("Training Logistic Regression model...")
    model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    joblib.dump(model, model_output_path) # Save the trained model
    print(f"Model trained and saved to {model_output_path}")
    time.sleep(1) # Simulate work

if __name__ == "__main__":
    train_model()
