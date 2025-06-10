import pandas as pd
from sklearn.model_selection import train_test_split

print("Running data_preprocessing.py...")

# Load data (assuming input.csv is in the 'data' directory relative to the project root)
try:
    df = pd.read_csv('data/input.csv')
    print("input.csv loaded successfully.")
except FileNotFoundError:
    print("Error: input.csv not found. Make sure it's in the 'data/' directory.")
    exit(1)

# Basic preprocessing (e.g., one-hot encode categorical features if any)
# For this simple dataset, let's assume no complex preprocessing needed beyond splitting.
# If 'location' or 'is_amex' were truly categorical strings, you'd convert them here.
# For now, treat them as numerical as per the input.csv structure.

# Define features (X) and target (y)
X = df[['amount', 'amount_per_location', 'location', 'is_amex']]
y = df['is_fraud']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42, stratify=y) # Small test_size for quick generation

# Save processed data (relative to the project root)
# Ensure the 'data' directory exists
import os
os.makedirs('data', exist_ok=True) # Create data directory if it doesn't exist

X_train.to_csv('data/processed_train_data.csv', index=False)
X_test.to_csv('data/processed_test_data.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False) # Ensure y_test is saved for evaluation and prediction testing

print(f"Data preprocessing complete. Saved train/test data to 'data/' directory.")
print(f"Train data shape: {X_train.shape}, Test data shape: {X_test.shape}")
