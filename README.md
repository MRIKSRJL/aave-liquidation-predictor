# Aave DeFi Liquidation Predictor 

An end-to-end Machine Learning pipeline designed to predict wallet liquidations on the Aave protocol (Polygon network). This project extracts raw blockchain data, performs financial feature engineering using live market prices, and trains an XGBoost model optimized for highly imbalanced classes.

## Project Overview

In Decentralized Finance (DeFi), liquidations occur when a borrower's collateral value falls below a certain threshold relative to their debt. Predicting these events is a classic quantitative finance problem, complicated by extreme class imbalance (healthy accounts vastly outnumber liquidated ones) and temporal state biases.

This project demonstrates a complete Data Engineering and Data Science workflow:
1. **Data Extraction**: Querying decentralized databases (The Graph) using GraphQL.
2. **Feature Engineering**: Transforming raw token balances into USD-denominated financial metrics (LTV, Total Collateral, Total Debt) using the CoinGecko API.
3. **Machine Learning**: Building a predictive model using XGBoost, handling the extreme class imbalance (`scale_pos_weight`), and evaluating risk-specific metrics (Precision/Recall on the minority class).

## 🏗️ Architecture & Pipeline

### 1. Data Extraction (`extract.py` & `extract_liquidated.py`)
- Interfaces with the **Aave Polygon Subgraph** (Messari standard).
- Implements robust pagination and retry mechanisms to bypass GraphQL timeout limits.
- Extracts two separate cohorts to counter class imbalance:
  - A random sample of active accounts (mostly healthy).
  - A targeted sample of specific `liquidatee` events to build the minority class.

### 2. Feature Engineering (`feature_engineering.py`)
- Merges healthy and liquidated datasets while avoiding data leakage.
- Normalizes token balances using blockchain decimals (e.g., 18 for WETH, 8 for WBTC).
- Integrates the **CoinGecko API** to fetch live USD prices.
- Calculates critical financial indicators:
  - `total_collateral_usd`: Total value of deposited assets.
  - `total_debt_usd`: Total value of borrowed assets.
  - `ltv` (Loan-to-Value): The primary risk indicator.

### 3. Model Training (`train_model.py`)
- Utilizes **XGBoost** (`XGBClassifier`) for tabular data prediction.
- Dynamically calculates the `scale_pos_weight` ratio to penalize the model for missing liquidations (handling the ~95% vs ~5% class imbalance).
- Outputs a comprehensive Confusion Matrix and Classification Report to analyze false positives (paranoia) vs. false negatives (missed risk).

## Installation & Usage

### Prerequisites
- Python 3.8+
- Required libraries: `pandas`, `xgboost`, `scikit-learn`, `requests`

```bash
pip install pandas xgboost scikit-learn requests