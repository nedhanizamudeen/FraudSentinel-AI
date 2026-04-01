import pandas as pd
import numpy as np
import os
import json
from config import (
    TRANSACTION_FILE, IDENTITY_FILE,
    TARGET_COLUMN, MAX_MISSING_RATIO
)


# ══════════════════════════════════════════
#  TOOL 1 — Load and Merge Dataset
# ══════════════════════════════════════════
def load_and_merge_data() -> dict:
    """
    Loads train_transaction.csv and train_identity.csv
    and merges them into one big DataFrame.

    WHY MERGE?
    The IEEE-CIS dataset comes in two separate files:
      - train_transaction.csv → has the money/card details
      - train_identity.csv    → has device/browser details
    We need both together to detect fraud properly.
    They are joined using the 'TransactionID' column.

    Returns a dict with:
      - 'data'    : the merged DataFrame
      - 'summary' : a text summary of the data
      - 'status'  : 'success' or 'error'
    """
    try:
        print("  [DataTool] Loading transaction data...")
        transactions = pd.read_csv(TRANSACTION_FILE)

        print("  [DataTool] Loading identity data...")
        identity = pd.read_csv(IDENTITY_FILE)

        print("  [DataTool] Merging on TransactionID...")
        # Left join: keep all transactions, add identity info where available
        df = transactions.merge(identity, on="TransactionID", how="left")

        summary = (
            f"Dataset loaded successfully.\n"
            f"  Rows         : {df.shape[0]:,}\n"
            f"  Columns      : {df.shape[1]}\n"
            f"  Fraud cases  : {df[TARGET_COLUMN].sum():,} "
            f"({df[TARGET_COLUMN].mean()*100:.2f}%)\n"
            f"  Non-fraud    : {(df[TARGET_COLUMN]==0).sum():,} "
            f"({(1-df[TARGET_COLUMN].mean())*100:.2f}%)\n"
            f"  Memory usage : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB"
        )

        print(f"  [DataTool] {summary}")
        return {"data": df, "summary": summary, "status": "success"}

    except FileNotFoundError as e:
        msg = (
            f"ERROR: Dataset file not found.\n"
            f"Please download IEEE-CIS Fraud Detection dataset from Kaggle\n"
            f"and place CSV files in the /data folder.\n"
            f"Missing file: {e.filename}"
        )
        return {"data": None, "summary": msg, "status": "error"}


# ══════════════════════════════════════════
#  TOOL 2 — Analyze Missing Values
# ══════════════════════════════════════════
def analyze_missing_values(df: pd.DataFrame) -> dict:
    """
    Finds all columns with missing values and reports them.

    WHY THIS MATTERS:
    IEEE-CIS dataset has 400+ columns and many have >50% missing values.
    Feeding missing values into ML models causes errors or bad predictions.
    This tool tells the agent what needs to be cleaned.

    Returns a dict with:
      - 'missing_report' : DataFrame of columns + their missing %
      - 'columns_to_drop': list of columns where >50% values are missing
      - 'summary'        : human-readable text summary
    """
    total_rows = len(df)

    # Calculate missing % for each column
    missing = df.isnull().sum()
    missing_pct = (missing / total_rows * 100).round(2)

    missing_report = pd.DataFrame({
        "column"      : missing.index,
        "missing_count": missing.values,
        "missing_pct" : missing_pct.values
    })
    missing_report = missing_report[missing_report["missing_count"] > 0]
    missing_report = missing_report.sort_values("missing_pct", ascending=False)

    # Columns with too many missing values — not useful for model
    columns_to_drop = list(
        missing_report[missing_report["missing_pct"] > MAX_MISSING_RATIO * 100]["column"]
    )

    summary = (
        f"Missing Value Analysis:\n"
        f"  Total columns with missing values : {len(missing_report)}\n"
        f"  Columns to DROP (>{MAX_MISSING_RATIO*100:.0f}% missing) : {len(columns_to_drop)}\n"
        f"  Top 5 most incomplete columns:\n"
    )
    for _, row in missing_report.head(5).iterrows():
        summary += f"    - {row['column']}: {row['missing_pct']:.1f}% missing\n"

    return {
        "missing_report"  : missing_report,
        "columns_to_drop" : columns_to_drop,
        "summary"         : summary
    }


# ══════════════════════════════════════════
#  TOOL 3 — Analyze Class Imbalance
# ══════════════════════════════════════════
def analyze_class_balance(df: pd.DataFrame) -> dict:
    """
    Checks how imbalanced the fraud vs non-fraud classes are.

    WHY THIS MATTERS:
    In IEEE-CIS, only ~3.5% of transactions are fraud.
    If we train a model on this directly, it will just predict
    'not fraud' for everything and get 96.5% accuracy — which is USELESS.
    The agent needs to know this to decide how to fix it (using SMOTE).

    Returns a dict with:
      - 'fraud_count'    : number of fraud transactions
      - 'non_fraud_count': number of non-fraud transactions
      - 'fraud_ratio'    : percentage of fraud
      - 'is_imbalanced'  : True if ratio < 20% (needs fixing)
      - 'summary'        : human-readable summary
    """
    fraud_count     = int(df[TARGET_COLUMN].sum())
    non_fraud_count = int((df[TARGET_COLUMN] == 0).sum())
    total           = fraud_count + non_fraud_count
    fraud_ratio     = fraud_count / total * 100

    is_imbalanced = fraud_ratio < 20  # if fraud < 20%, we have imbalance

    summary = (
        f"Class Balance Analysis:\n"
        f"  Fraud transactions     : {fraud_count:,} ({fraud_ratio:.2f}%)\n"
        f"  Non-fraud transactions : {non_fraud_count:,} ({100-fraud_ratio:.2f}%)\n"
        f"  Imbalance ratio        : 1:{int(non_fraud_count/fraud_count)}\n"
        f"  Needs SMOTE fix?       : {'YES' if is_imbalanced else 'NO'}\n"
    )

    if is_imbalanced:
        summary += (
            f"  RECOMMENDATION: Apply SMOTE to oversample minority class.\n"
            f"  This will create synthetic fraud examples to balance the dataset."
        )

    return {
        "fraud_count"    : fraud_count,
        "non_fraud_count": non_fraud_count,
        "fraud_ratio"    : fraud_ratio,
        "is_imbalanced"  : is_imbalanced,
        "summary"        : summary
    }


# ══════════════════════════════════════════
#  TOOL 4 — Get Column Types
# ══════════════════════════════════════════
def get_column_types(df: pd.DataFrame) -> dict:
    """
    Separates columns into numerical and categorical types.

    WHY THIS MATTERS:
    ML models need numbers. Categorical columns (like card type = 'Visa')
    need to be converted to numbers first. This tool identifies which
    columns need that conversion.

    Returns:
      - 'numerical_cols'   : list of numeric column names
      - 'categorical_cols' : list of object/string column names
      - 'summary'          : text summary
    """
    # Exclude the target column and ID
    exclude = [TARGET_COLUMN, "TransactionID"]

    numerical_cols   = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c not in exclude]
    categorical_cols = [c for c in df.select_dtypes(include=["object"]).columns
                        if c not in exclude]

    summary = (
        f"Column Type Analysis:\n"
        f"  Numerical columns   : {len(numerical_cols)}\n"
        f"  Categorical columns : {len(categorical_cols)}\n"
        f"  Sample categorical  : {categorical_cols[:5]}\n"
    )

    return {
        "numerical_cols"   : numerical_cols,
        "categorical_cols" : categorical_cols,
        "summary"          : summary
    }


# ══════════════════════════════════════════
#  TOOL 5 — Save Report to File
# ══════════════════════════════════════════
def save_report(report_name: str, content: str, reports_dir: str) -> str:
    """
    Saves a text report to the /reports folder.
    Every agent saves its findings here for full audit trail.
    """
    os.makedirs(reports_dir, exist_ok=True)
    filepath = os.path.join(reports_dir, f"{report_name}.txt")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filepath
