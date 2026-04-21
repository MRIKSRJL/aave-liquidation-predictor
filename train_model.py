import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import os

# Suppress non-actionable warnings in notebook-style runs.
warnings.filterwarnings('ignore')

INPUT_FILE = "aave_dataset_advanced.csv"

def train_xgboost():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")

    print(f"Loading dataset from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # 1. Financial features in USD
    X = df[['total_collateral_usd', 'total_debt_usd', 'ltv', 'num_positions']]
    y = df['is_liquidated']

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Class imbalance compensation weight
    class_counts = y_train.value_counts()
    negative_count = class_counts.get(0, 0)
    positive_count = class_counts.get(1, 0)
    if positive_count == 0:
        raise ValueError("Training split contains no positive samples (class 1).")
    ratio = negative_count / positive_count
    print(f"\nComputed scale_pos_weight: {ratio:.2f}")

    # 4. XGBoost model configuration
    print("\nTraining XGBoost classifier...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=ratio,
        max_depth=5,
        learning_rate=0.1,
        n_estimators=150,
        random_state=42
    )

    # 5. Model fitting
    model.fit(X_train, y_train)
    print("Training complete.")

    # 6. Evaluation inference
    y_pred = model.predict(X_test)

    # 7. Evaluation report
    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS (FINANCIAL FEATURES)")
    print("=" * 50)

    print("\n1. Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negatives (0 predicted as 0): {cm[0][0]}")
    print(f"False Positives (0 predicted as 1): {cm[0][1]}")
    print(f"False Negatives (1 predicted as 0): {cm[1][0]}")
    print(f"True Positives (1 predicted as 1): {cm[1][1]}")

    print("\n2. Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_xgboost()