import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

from config import (
    TARGET_COLUMN, TEST_SIZE, RANDOM_STATE,
    MODELS_DIR, MAX_MISSING_RATIO
)


# ══════════════════════════════════════════
#  TOOL 1 — Clean and Prepare Data
# ══════════════════════════════════════════
def clean_data(df: pd.DataFrame, columns_to_drop: list) -> dict:
    """
    Cleans the raw merged dataframe:
      1. Drops columns with too many missing values
      2. Drops the TransactionID (not useful for prediction)
      3. Fills remaining missing values with median (for numbers)
         and 'Unknown' (for text columns)

    WHY FILL MISSING VALUES?
    XGBoost can actually handle NaN values internally,
    but LightGBM and SMOTE cannot. We fill them to be safe.

    Returns: cleaned DataFrame + summary
    """
    print("  [MLTool] Dropping high-missing columns...")
    df_clean = df.drop(columns=columns_to_drop + ["TransactionID"],
                       errors="ignore")

    # Fill numerical missing values with median
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COLUMN in num_cols:
        num_cols.remove(TARGET_COLUMN)

    for col in num_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Fill categorical missing values with 'Unknown'
    cat_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        df_clean[col] = df_clean[col].fillna("Unknown")

    summary = (
        f"Data Cleaning Complete:\n"
        f"  Dropped {len(columns_to_drop)} high-missing columns\n"
        f"  Filled {len(num_cols)} numerical columns with median\n"
        f"  Filled {len(cat_cols)} categorical columns with 'Unknown'\n"
        f"  Final shape: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} columns"
    )

    return {"data": df_clean, "summary": summary}


# ══════════════════════════════════════════
#  TOOL 2 — Encode Categorical Columns
# ══════════════════════════════════════════
def encode_categoricals(df: pd.DataFrame) -> dict:
    """
    Converts text columns to numbers using Label Encoding.

    EXAMPLE:
      card4 = ['visa', 'mastercard', 'visa', 'amex']
         → card4 = [2, 1, 2, 0]  (numbers the model understands)

    WHY LABEL ENCODING (not One-Hot)?
    With 400+ columns, One-Hot encoding would create thousands
    of new columns and make the model very slow. Label Encoding
    keeps it efficient.

    Returns: encoded DataFrame + list of encoded columns + encoders dict
    """
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}

    print(f"  [MLTool] Encoding {len(cat_cols)} categorical columns...")

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le  # save encoder so we can decode later

    summary = (
        f"Categorical Encoding Complete:\n"
        f"  Encoded {len(cat_cols)} columns using Label Encoding\n"
        f"  Sample columns: {cat_cols[:5]}"
    )

    return {
        "data"             : df,
        "encoded_columns"  : cat_cols,
        "encoders"         : encoders,
        "summary"          : summary
    }


# ══════════════════════════════════════════
#  TOOL 3 — Engineer New Features
# ══════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> dict:
    """
    Creates new features that help detect fraud better.

    WHAT IS FEATURE ENGINEERING?
    The raw data has columns like TransactionAmt, card1, addr1, etc.
    These alone are not very informative. But when we COMBINE them
    or CREATE RATIOS, patterns become clearer.

    Example: A $5000 transaction might not be suspicious on its own.
    But if the AVERAGE transaction for that card is $50, then
    $5000 / $50 = 100x the average → very suspicious!

    New features created:
      1. TransactionAmt_log        → log scale (reduces outlier effect)
      2. TransactionAmt_to_mean    → how far from card's average amount
      3. hour_of_transaction       → what hour did it happen
      4. is_night_transaction      → was it between midnight and 6AM?
      5. amt_addr_ratio            → transaction amount per address group
    """
    print("  [MLTool] Engineering new features...")
    new_features = []
    new_cols = {}   # collect all new columns first, then assign at once

    # Feature 1: Log of transaction amount
    if "TransactionAmt" in df.columns:
        new_cols["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
        new_features.append("TransactionAmt_log")

    # Feature 2: Amount relative to card's average
    if all(c in df.columns for c in ["TransactionAmt", "card1"]):
        card_mean = df.groupby("card1")["TransactionAmt"].transform("mean")
        new_cols["TransactionAmt_to_card_mean"] = df["TransactionAmt"] / (card_mean + 1)
        new_features.append("TransactionAmt_to_card_mean")

    # Feature 3: Hour and day of transaction
    if "TransactionDT" in df.columns:
        new_cols["transaction_hour"] = (df["TransactionDT"] // 3600) % 24
        new_cols["transaction_day"]  = (df["TransactionDT"] // 86400) % 7
        new_features.extend(["transaction_hour", "transaction_day"])

    # Feature 4: Night transaction flag
    if "TransactionDT" in df.columns:
        hour = (df["TransactionDT"] // 3600) % 24
        new_cols["is_night_transaction"] = ((hour >= 0) & (hour <= 5)).astype(int)
        new_features.append("is_night_transaction")

    # Feature 5: Transaction count per card
    if "card1" in df.columns:
        new_cols["card1_count"] = df.groupby("card1")["card1"].transform("count")
        new_features.append("card1_count")

    # Feature 6: Transaction amount per address group
    if all(c in df.columns for c in ["TransactionAmt", "addr1"]):
        addr_mean = df.groupby("addr1")["TransactionAmt"].transform("mean")
        new_cols["amt_addr_ratio"] = df["TransactionAmt"] / (addr_mean + 1)
        new_features.append("amt_addr_ratio")

    # Assign all new columns at once — fixes PerformanceWarning
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    df = df.copy()  # defragment the DataFrame

    summary = (
        f"Feature Engineering Complete:\n"
        f"  Created {len(new_features)} new features:\n"
    )
    for f in new_features:
        summary += f"    + {f}\n"

    return {"data": df, "new_features": new_features, "summary": summary}


# ══════════════════════════════════════════
#  TOOL 4 — Split Data for Training
# ══════════════════════════════════════════
def split_data(df: pd.DataFrame) -> dict:
    """
    Splits data into:
      - X_train : features for training (80%)
      - X_test  : features for testing (20%)
      - y_train : fraud labels for training
      - y_test  : fraud labels for testing

    WHY SPLIT?
    We train the model on 80% of data, then TEST it on the
    remaining 20% it has NEVER seen before.
    This tells us how well it will work on real new transactions.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # ensures same fraud % in both train and test sets
    )

    summary = (
        f"Data Split Complete:\n"
        f"  Training set : {X_train.shape[0]:,} rows ({(1-TEST_SIZE)*100:.0f}%)\n"
        f"  Testing set  : {X_test.shape[0]:,} rows ({TEST_SIZE*100:.0f}%)\n"
        f"  Features     : {X_train.shape[1]} columns\n"
        f"  Fraud in train: {y_train.sum():,} ({y_train.mean()*100:.2f}%)"
    )

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_names": list(X.columns),
        "summary": summary
    }


# ══════════════════════════════════════════
#  TOOL 5 — Apply SMOTE (Fix Imbalance)
# ══════════════════════════════════════════
def apply_smote(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Applies SMOTE (Synthetic Minority Oversampling Technique).

    HOW SMOTE WORKS:
    Instead of just duplicating fraud examples (which causes overfitting),
    SMOTE creates SYNTHETIC (fake but realistic) fraud transactions
    by interpolating between existing fraud examples.

    Example:
      Before SMOTE: 500,000 non-fraud, 18,000 fraud
      After  SMOTE: 500,000 non-fraud, 500,000 synthetic-fraud
      Now the model trains on balanced data!

    NOTE: SMOTE is applied ONLY to training data, never to test data.
    We want to evaluate on real, imbalanced data to simulate real world.
    """
    print("  [MLTool] Applying SMOTE to balance training data...")
    print(f"    Before: {y_train.value_counts().to_dict()}")

    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"    After : {pd.Series(y_resampled).value_counts().to_dict()}")

    summary = (
        f"SMOTE Applied:\n"
        f"  Before: {y_train.sum():,} fraud / {(y_train==0).sum():,} non-fraud\n"
        f"  After : {y_resampled.sum():,} fraud / {(y_resampled==0).sum():,} non-fraud\n"
        f"  Training set now perfectly balanced!"
    )

    return {
        "X_train_resampled": X_resampled,
        "y_train_resampled": y_resampled,
        "summary": summary
    }


# ══════════════════════════════════════════
#  TOOL 6 — Train XGBoost Model
# ══════════════════════════════════════════
def train_xgboost(X_train, y_train) -> dict:
    """
    Trains an XGBoost classifier.

    WHAT IS XGBOOST?
    XGBoost = eXtreme Gradient Boosting
    It builds many decision trees one by one.
    Each new tree CORRECTS the mistakes of the previous ones.
    The final model is the combined wisdom of all trees.
    It is one of the most powerful algorithms for tabular data.

    Key parameters:
      n_estimators  : how many trees to build (200)
      max_depth     : how deep each tree grows (6)
      learning_rate : how much each tree contributes (0.05, smaller = better)
      scale_pos_weight: handles class imbalance internally
    """
    print("  [MLTool] Training XGBoost model...")

    fraud_count = int((y_train == 1).sum())
    non_fraud   = int((y_train == 0).sum())
    scale_weight = non_fraud / fraud_count if fraud_count > 0 else 1

    model = xgb.XGBClassifier(
        n_estimators      = 200,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = scale_weight,
        use_label_encoder = False,
        eval_metric       = "auc",
        random_state      = RANDOM_STATE,
        n_jobs            = -1,      # use all CPU cores
        verbosity         = 0
    )

    model.fit(X_train, y_train)

    # Save the model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    joblib.dump(model, model_path)

    return {
        "model"      : model,
        "model_name" : "XGBoost",
        "model_path" : model_path,
        "summary"    : "XGBoost model trained and saved successfully."
    }


# ══════════════════════════════════════════
#  TOOL 7 — Train LightGBM Model
# ══════════════════════════════════════════
def train_lightgbm(X_train, y_train) -> dict:
    """
    Trains a LightGBM classifier.

    WHAT IS LIGHTGBM?
    LightGBM is similar to XGBoost but:
      - FASTER (uses histogram-based splitting)
      - Uses LESS memory (great for large datasets like IEEE-CIS)
      - Often gives BETTER AUC scores
    It is Microsoft's open-source boosting framework.

    The agent will compare XGBoost vs LightGBM and pick the winner.
    """
    print("  [MLTool] Training LightGBM model...")

    model = lgb.LGBMClassifier(
        n_estimators   = 200,
        max_depth       = 6,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        class_weight    = "balanced",
        random_state    = RANDOM_STATE,
        n_jobs          = -1,
        verbose         = -1     # suppress training logs
    )

    model.fit(X_train, y_train)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "lightgbm_model.pkl")
    joblib.dump(model, model_path)

    return {
        "model"      : model,
        "model_name" : "LightGBM",
        "model_path" : model_path,
        "summary"    : "LightGBM model trained and saved successfully."
    }


# ══════════════════════════════════════════
#  TOOL 8 — Evaluate Model Performance
# ══════════════════════════════════════════
def evaluate_model(model, model_name: str, X_test, y_test) -> dict:
    """
    Evaluates how well the trained model performs.

    METRICS EXPLAINED:
      AUC-ROC  → The main metric. Measures ability to distinguish
                  fraud from non-fraud. 1.0 = perfect, 0.5 = random.
                  We target AUC > 0.85.

      Precision → Of all transactions we flagged as fraud,
                  how many were ACTUALLY fraud?
                  Low precision = many false alarms (annoying customers)

      Recall    → Of all ACTUAL fraud transactions,
                  how many did we CATCH?
                  Low recall = missing real fraud (dangerous!)

      F1 Score  → Balance between Precision and Recall.
                  The harmonic mean of both.

    In fraud detection, RECALL is usually more important than Precision.
    Missing real fraud is worse than flagging a legitimate transaction.
    """
    print(f"  [MLTool] Evaluating {model_name}...")

    # Get fraud probability (not just 0/1 prediction)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    auc       = roc_auc_score(y_test, y_proba)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    summary = (
        f"Evaluation Results — {model_name}:\n"
        f"  AUC-ROC   : {auc:.4f}  {'✓ GOOD' if auc >= 0.85 else '✗ NEEDS IMPROVEMENT'}\n"
        f"  F1 Score  : {f1:.4f}\n"
        f"  Precision : {precision:.4f}\n"
        f"  Recall    : {recall:.4f}\n"
        f"  ──────────────────────\n"
        f"  Confusion Matrix:\n"
        f"    True Negatives  (correct non-fraud) : {tn:,}\n"
        f"    False Positives (wrong fraud alarm)  : {fp:,}\n"
        f"    False Negatives (missed real fraud)  : {fn:,}\n"
        f"    True Positives  (caught fraud)       : {tp:,}\n"
    )

    return {
        "model_name" : model_name,
        "auc"        : auc,
        "f1"         : f1,
        "precision"  : precision,
        "recall"     : recall,
        "y_proba"    : y_proba,
        "confusion_matrix": cm,
        "summary"    : summary
    }


# ══════════════════════════════════════════
#  TOOL 9 — Predict on Single Transaction
# ══════════════════════════════════════════
def predict_transaction(model, transaction: dict, feature_names: list) -> dict:
    """
    Takes a single new transaction and predicts fraud probability.

    This is what the Decision Agent uses to flag individual transactions.

    Returns:
      - fraud_probability : float between 0.0 and 1.0
      - is_fraud          : True/False
      - risk_level        : LOW / MEDIUM / HIGH / CRITICAL
    """
    # Convert dict to DataFrame with correct column order
    row = pd.DataFrame([transaction])

    # Add any missing columns as 0
    for col in feature_names:
        if col not in row.columns:
            row[col] = 0

    row = row[feature_names]  # ensure same column order as training

    fraud_prob = model.predict_proba(row)[0][1]

    # Risk level classification
    if fraud_prob < 0.3:
        risk_level = "LOW"
    elif fraud_prob < 0.5:
        risk_level = "MEDIUM"
    elif fraud_prob < 0.75:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    return {
        "fraud_probability": round(float(fraud_prob), 4),
        "is_fraud"         : fraud_prob >= 0.5,
        "risk_level"       : risk_level
    }
