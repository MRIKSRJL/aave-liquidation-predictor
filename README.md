# Aave DeFi Liquidation Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Model](https://img.shields.io/badge/Model-XGBoost-orange)
![Domain](https://img.shields.io/badge/Domain-DeFi%20Risk-purple)
![PR--AUC](https://img.shields.io/badge/PR--AUC-0.0886-lightgrey)
![Class1 Recall](https://img.shields.io/badge/Class1%20Recall-0.20-lightgrey)
![Security](https://img.shields.io/badge/Security-Secrets%20Hygiene-success)

End-to-end Machine Learning pipeline to detect liquidation risk on Aave (Polygon) from on-chain data.  
The project combines secure data extraction, financial feature engineering, and imbalanced classification with XGBoost.

## TL;DR

- Built a secure ELT + ML pipeline for DeFi liquidation-risk detection on Aave (Polygon).
- Added production-minded extraction controls: timeout, retry/backoff, GraphQL error handling, pagination safety, and incremental persistence.
- Engineered USD risk features and concentration/composition proxies from on-chain positions.
- Evaluated imbalanced classification with PR-AUC, threshold sweeps, and cost-sensitive decision rules.
- Latest holdout baseline: PR-AUC `0.0886` with explicit minority-class trade-off analysis.

## Why This Project

Liquidations in DeFi are rare but high-impact events. This creates a realistic quantitative finance challenge:

- Severe class imbalance (few liquidations vs many healthy accounts)
- Noisy on-chain state
- Operational trade-off between false alarms and missed liquidations

This repository is designed as a production-minded portfolio project with explicit security controls and robust data collection.

## Pipeline Overview

### 1) Data Extraction

- `extract.py`: collects account cohorts from The Graph subgraph
- `extract_liquidated.py`: collects targeted liquidated accounts (`liquidatee`)
- Implements:
  - retry with backoff
  - request timeouts
  - GraphQL error handling (including HTTP 200 + `errors`)
  - pagination safeguard against non-advancing cursors
  - incremental `.jsonl` persistence to reduce crash data loss

### 2) Feature Engineering

- `feature_engineering.py` merges healthy and liquidated cohorts
- Converts token balances to USD using CoinGecko prices
- Core features:
  - `total_collateral_usd`
  - `total_debt_usd`
  - `ltv`
  - `num_positions`
- Additional risk proxy features:
  - `num_lend_positions`
  - `num_borrow_positions`
  - `num_debt_assets`
  - `debt_stable_share`
  - `collateral_volatile_share`
  - `debt_concentration_hhi`
  - `collateral_concentration_hhi`

### 3) Model Training and Evaluation

- `train_model.py` trains XGBoost for binary classification (`is_liquidated`)
- Handles imbalance with dynamic `scale_pos_weight`
- Evaluates:
  - confusion matrix
  - classification report
  - PR-AUC (Average Precision)
  - threshold sweep (precision/recall trade-off)
  - business-cost thresholding (`FN` weighted more than `FP`)
- Compares baseline training vs controlled undersampling

## Project Structure

```text
.
|- extract.py
|- extract_liquidated.py
|- feature_engineering.py
|- train_model.py
|- requirements.txt
|- .env.example
|- .gitignore
|- README.md
```

## Setup

### Prerequisites

- Python 3.8+
- Git (optional, for version control)

### Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows PowerShell
pip install -r requirements.txt
```

## Environment Variables

Create a local `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Set:

```env
GRAPH_API_KEY=your_graph_api_key_here
```

## Run the Pipeline

Execute in this order:

```bash
python extract.py
python extract_liquidated.py
python feature_engineering.py
python train_model.py
```

Expected generated artifacts (local only, ignored by Git):

- `aave_raw_users.jsonl`
- `aave_liquidated_users.jsonl`
- `aave_dataset_advanced.csv`

## Current Results (Latest Run)

Dataset:

- Rows: `4,997`
- Features used by model: `11`
- Positive class support in test split: `41 / 1000`

Model comparison (holdout):

- Baseline PR-AUC: `0.0886`
- Undersampled PR-AUC: `0.0674`
- Selected model: `Baseline`

Baseline confusion matrix at threshold `0.50`:

- True Negatives: `846`
- False Positives: `113`
- False Negatives: `33`
- True Positives: `8`

Business-cost thresholding example (`FN cost = 5`, `FP cost = 1`):

- Cost-optimized threshold from tested grid: `0.70`
- Cost: `233`
- Precision (class 1): `0.065`
- Recall (class 1): `0.073`

## Security Notes

- `.env` is excluded by `.gitignore`
- `.env.example` contains placeholders only
- No API keys should ever be committed
- Data artifacts (`.csv`, `.jsonl`, model binaries) are ignored
- If a key is exposed, rotate/revoke it immediately in The Graph dashboard

Pre-push checks:

```bash
git ls-files .env
git ls-files "*.csv" "*.jsonl" "*.pkl" "*.model"
```

Both commands should return no tracked secret/data artifacts.

## Limitations

- Current predictive signal for minority class remains weak (low precision/recall for class 1)
- Uses snapshot-style pricing rather than fully time-aligned historical market context
- Not production trading infrastructure and not financial advice

## Next Improvements

- Add temporal and market-regime features
- Introduce probability calibration and threshold policies by risk appetite
- Evaluate alternative imbalance strategies (focal loss, ensemble methods)
- Add experiment tracking and reproducible model artifacts

## Recruiter Notes

This project demonstrates:

- Data extraction reliability (timeouts, retries, pagination guards)
- Security hygiene for public repositories
- Feature engineering on blockchain financial state
- Imbalanced learning evaluation beyond accuracy (PR-AUC, threshold economics)

It is intended as a practical, security-aware ML engineering portfolio case study.