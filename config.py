import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "paste_your_groq_key_here")

LLM_MODEL = "llama-3.3-70b-versatile"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRANSACTION_FILE = os.path.join(DATA_DIR, "train_transaction.csv")
IDENTITY_FILE    = os.path.join(DATA_DIR, "train_identity.csv")
TEST_TRANSACTION = os.path.join(DATA_DIR, "test_transaction.csv")
TEST_IDENTITY    = os.path.join(DATA_DIR, "test_identity.csv")

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

#ML SETTINGS: These control how the ML models train

RANDOM_STATE  = 42        # for reproducibility (same results every run)
TEST_SIZE     = 0.2       # 20% data used for testing, 80% for training
TARGET_COLUMN = "isFraud" # the column we want to predict

# Minimum AUC score the Eval Agent will accept.
# If the model scores below this, it loops back and retrains.
MIN_AUC_THRESHOLD = 0.85

#  6. FRAUD DECISION THRESHOLD: If fraud probability > this value → FLAG as fraud
FRAUD_THRESHOLD = 0.5

#  7. FEATURE ENGINEERING SETTINGS: Columns with more than this % missing values are dropped
MAX_MISSING_RATIO = 0.5   # drop column if >50% values are missing

#  8. DISPLAY SETTINGS
PROJECT_NAME    = "FraudSentinel AI"
PROJECT_VERSION = "1.0.0"