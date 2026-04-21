import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
import warnings
import os

# Suppress non-actionable warnings in notebook-style runs.
warnings.filterwarnings('ignore')

INPUT_FILE = "aave_dataset_advanced.csv"
THRESHOLD_CANDIDATES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]
RANDOM_STATE = 42
UNDERSAMPLE_NEG_TO_POS_RATIO = 3
FALSE_NEGATIVE_COST = 5
FALSE_POSITIVE_COST = 1
FEATURE_COLUMNS = [
    "total_collateral_usd",
    "total_debt_usd",
    "ltv",
    "num_positions",
    "num_lend_positions",
    "num_borrow_positions",
    "num_debt_assets",
    "debt_stable_share",
    "collateral_volatile_share",
    "debt_concentration_hhi",
    "collateral_concentration_hhi",
]


def build_model(scale_pos_weight):
    return xgb.XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        max_depth=5,
        learning_rate=0.1,
        n_estimators=150,
        random_state=RANDOM_STATE,
    )


def build_undersampled_training_set(X_train, y_train):
    train_df = X_train.copy()
    train_df["is_liquidated"] = y_train.values

    positives = train_df[train_df["is_liquidated"] == 1]
    negatives = train_df[train_df["is_liquidated"] == 0]

    if positives.empty:
        raise ValueError("Cannot undersample because there are no positive samples.")

    target_negative_count = min(len(negatives), len(positives) * UNDERSAMPLE_NEG_TO_POS_RATIO)
    sampled_negatives = negatives.sample(n=target_negative_count, random_state=RANDOM_STATE)

    balanced_df = pd.concat([positives, sampled_negatives]).sample(
        frac=1.0, random_state=RANDOM_STATE
    )
    X_balanced = balanced_df[FEATURE_COLUMNS]
    y_balanced = balanced_df["is_liquidated"]
    return X_balanced, y_balanced


def cross_validate_pr_auc(X, y, scale_pos_weight):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for train_idx, valid_idx in cv.split(X, y):
        X_fold_train = X.iloc[train_idx]
        y_fold_train = y.iloc[train_idx]
        X_fold_valid = X.iloc[valid_idx]
        y_fold_valid = y.iloc[valid_idx]

        model = build_model(scale_pos_weight=scale_pos_weight)
        model.fit(X_fold_train, y_fold_train)
        y_fold_proba = model.predict_proba(X_fold_valid)[:, 1]
        scores.append(average_precision_score(y_fold_valid, y_fold_proba))

    return sum(scores) / len(scores)


def evaluate_model(model_name, model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.50).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_proba)

    print("\n" + "=" * 50)
    print(f"{model_name} - EVALUATION RESULTS")
    print("=" * 50)
    print("\n1. Confusion Matrix:")
    print(f"True Negatives (0 predicted as 0): {cm[0][0]}")
    print(f"False Positives (0 predicted as 1): {cm[0][1]}")
    print(f"False Negatives (1 predicted as 0): {cm[1][0]}")
    print(f"True Positives (1 predicted as 1): {cm[1][1]}")

    print("\n2. Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"3. PR-AUC (Average Precision): {pr_auc:.4f}")

    return {"pr_auc": pr_auc, "y_proba": y_proba}


def threshold_sweep(y_test, y_proba):
    print("\n4. Threshold Sweep (minority-class trade-off):")
    print("threshold | precision_1 | recall_1 | business_cost")
    print("-" * 53)

    best_threshold = None
    best_precision = 0.0
    best_recall = -1.0
    best_cost_threshold = None
    best_cost_value = float("inf")
    best_cost_precision = 0.0
    best_cost_recall = 0.0
    best_cost_fp = 0
    best_cost_fn = 0

    for threshold in THRESHOLD_CANDIDATES:
        y_pred_threshold = (y_proba >= threshold).astype(int)
        precision_1 = precision_score(y_test, y_pred_threshold, zero_division=0)
        recall_1 = recall_score(y_test, y_pred_threshold, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()
        business_cost = (fn * FALSE_NEGATIVE_COST) + (fp * FALSE_POSITIVE_COST)
        print(
            f"{threshold:8.2f} | {precision_1:11.3f} | {recall_1:8.3f} | {business_cost:13.1f}"
        )

        if precision_1 >= 0.10 and recall_1 > best_recall:
            best_threshold = threshold
            best_precision = precision_1
            best_recall = recall_1

        if business_cost < best_cost_value:
            best_cost_value = business_cost
            best_cost_threshold = threshold
            best_cost_precision = precision_1
            best_cost_recall = recall_1
            best_cost_fp = int(fp)
            best_cost_fn = int(fn)

    if best_threshold is not None:
        print(
            "\nRecommended threshold for risk screening "
            f"(precision >= 0.10): {best_threshold:.2f} "
            f"(precision={best_precision:.3f}, recall={best_recall:.3f})"
        )
    else:
        print(
            "\nNo threshold reached precision >= 0.10. "
            "Collect more positive samples or add temporal features."
        )

    print(
        "\nCost-optimized threshold "
        f"(FN cost={FALSE_NEGATIVE_COST}, FP cost={FALSE_POSITIVE_COST}): "
        f"{best_cost_threshold:.2f} "
        f"(cost={best_cost_value:.1f}, precision={best_cost_precision:.3f}, "
        f"recall={best_cost_recall:.3f}, FP={best_cost_fp}, FN={best_cost_fn})"
    )


def train_xgboost():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")

    print(f"Loading dataset from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # 1. Financial and risk-composition features
    missing_features = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_features:
        raise ValueError(
            "Dataset is missing required engineered features. "
            f"Missing columns: {missing_features}. "
            "Run feature_engineering.py again."
        )

    X = df[FEATURE_COLUMNS]
    y = df["is_liquidated"]

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # 3. Class imbalance compensation weight
    class_counts = y_train.value_counts()
    negative_count = class_counts.get(0, 0)
    positive_count = class_counts.get(1, 0)
    if positive_count == 0:
        raise ValueError("Training split contains no positive samples (class 1).")
    ratio = negative_count / positive_count
    print(f"\nComputed scale_pos_weight: {ratio:.2f}")

    # 4. Cross-validation on training split with PR-AUC
    print("\nRunning 5-fold CV on baseline model (PR-AUC)...")
    baseline_cv_pr_auc = cross_validate_pr_auc(X_train, y_train, scale_pos_weight=ratio)
    print(f"Baseline CV PR-AUC: {baseline_cv_pr_auc:.4f}")

    # 5. Baseline training
    print("\nTraining baseline XGBoost model...")
    baseline_model = build_model(scale_pos_weight=ratio)
    baseline_model.fit(X_train, y_train)

    # 6. Undersampled training
    X_balanced, y_balanced = build_undersampled_training_set(X_train, y_train)
    print(
        "Training set after undersampling: "
        f"{len(y_balanced)} rows "
        f"(positives={int((y_balanced == 1).sum())}, negatives={int((y_balanced == 0).sum())})"
    )
    undersampled_model = build_model(scale_pos_weight=1.0)
    undersampled_model.fit(X_balanced, y_balanced)

    # 7. Holdout evaluation
    baseline_results = evaluate_model("BASELINE MODEL", baseline_model, X_test, y_test)
    undersampled_results = evaluate_model(
        "UNDERSAMPLED MODEL", undersampled_model, X_test, y_test
    )

    if undersampled_results["pr_auc"] > baseline_results["pr_auc"]:
        selected_name = "UNDERSAMPLED MODEL"
        selected_proba = undersampled_results["y_proba"]
    else:
        selected_name = "BASELINE MODEL"
        selected_proba = baseline_results["y_proba"]

    print(f"\nSelected model for threshold tuning: {selected_name}")
    threshold_sweep(y_test, selected_proba)

if __name__ == "__main__":
    train_xgboost()