import pandas as pd
import numpy as np
import os
import time

def generate_dummy_data(file_path="data/transactions.csv"):
    print("--- Data Preprocessing: Generating Dummy Data ---")
    if not os.path.exists('data'):
        os.makedirs('data')

    np.random.seed(42)
    num_samples = 2000
    data = {
        'transaction_id': range(num_samples),
        'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_samples, freq='H')),
        'amount': np.random.rand(num_samples) * 100 + 10,
        'location': np.random.randint(1, 10, num_samples),
        'card_type': np.random.choice(['visa', 'mastercard', 'amex'], num_samples),
        'merchant_id': np.random.randint(100, 200, num_samples),
        'is_fraud': np.random.choice([0, 1], num_samples, p=[0.98, 0.02]) # 2% fraud
    }
    df = pd.DataFrame(data)

    # Introduce some patterns for 'fraud'
    fraud_indices = df[df['is_fraud'] == 1].index
    df.loc[fraud_indices, 'amount'] = np.random.rand(len(fraud_indices)) * 1000 + 500 # High amounts
    df.loc[fraud_indices, 'location'] = np.random.randint(10, 15, len(fraud_indices)) # Unusual locations

    df.to_csv(file_path, index=False)
    print(f"Dummy data generated and saved to {file_path}")
    time.sleep(1) # Simulate work

def feature_engineering(input_path="data/transactions.csv", output_path="data/features.csv"):
    print("--- Data Preprocessing: Feature Engineering ---")
    df = pd.read_csv(input_path)

    # Simple feature engineering examples
    df['amount_per_location'] = df['amount'] / df.groupby('location')['amount'].transform('mean')
    df['is_amex'] = (df['card_type'] == 'amex').astype(int)

    features_df = df[['transaction_id', 'amount', 'amount_per_location', 'location', 'is_amex', 'is_fraud']]
    features_df.to_csv(output_path, index=False)
    print(f"Features engineered and saved to {output_path}")
    time.sleep(1) # Simulate work

if __name__ == "__main__":
    generate_dummy_data()
    feature_engineering()
