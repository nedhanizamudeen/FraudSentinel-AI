# 🛡️ FraudSentinel AI
### Autonomous Multi-Agent Fraud Detection System

---

## 📌 What is This Project?
FraudSentinel AI is a multi-agent autonomous pipeline built using
LangChain and LangGraph that detects financial fraud on the
IEEE-CIS dataset — without any human intervention.

**5 AI Agents:** EDA → Feature Engineering → Model Training →
Evaluation → Decision (with LLM explanation)

---

## ⚡ Setup Instructions for Teammates

### Step 1 — Clone the project
Open terminal and run:
```
git clone https://github.com/YOUR_USERNAME/FraudSentinel-AI.git
cd FraudSentinel-AI
```

### Step 2 — Install all libraries
```
pip install -r requirements.txt
```

### Step 3 — Get your FREE Groq API Key
1. Go to https://console.groq.com
2. Sign up with Google (free, no credit card needed)
3. Click **API Keys** in the left sidebar
4. Click **Create API Key** → copy the key

### Step 4 — Add your API key
Open `config.py` and find this line:
```python
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "paste_your_groq_key_here")
```
Replace `paste_your_groq_key_here` with your actual key.

### Step 5 — Download the Dataset
1. Go to https://www.kaggle.com/c/ieee-fraud-detection/data
2. Download `train_transaction.csv` and `train_identity.csv`
3. Place both files inside the `/data/` folder

### Step 6 — Run the project
```
python main.py
```

---

## 🗂️ Project Structure
```
FraudSentinel_AI/
├── agents/
│   ├── eda_agent.py          ← Agent 1: Data Analysis
│   ├── feature_agent.py      ← Agent 2: Feature Engineering
│   ├── model_agent.py        ← Agent 3: Model Training
│   ├── eval_agent.py         ← Agent 4: Evaluation + Loop
│   └── decision_agent.py     ← Agent 5: Fraud Decision
├── tools/
│   ├── data_tools.py         ← Data loading/cleaning functions
│   └── ml_tools.py           ← ML training/evaluation functions
├── orchestrator/
│   └── graph.py              ← LangGraph pipeline controller
├── data/                     ← Put Kaggle CSV files here
├── models/                   ← Saved models (auto-generated)
├── reports/                  ← Agent reports (auto-generated)
├── config.py                 ← Settings + API key goes here
├── main.py                   ← Run this file
└── requirements.txt          ← All dependencies
```

---

## 🤖 Tech Stack
| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Agentic Framework | LangChain + LangGraph |
| LLM Brain | LLaMA 3 via Groq API (FREE) |
| ML Models | XGBoost + LightGBM |
| Imbalance Fix | SMOTE + Tomek Links |
| Dataset | IEEE-CIS Fraud Detection (Kaggle) |

---

## 📊 Results
- **Best Model:** XGBoost
- **AUC-ROC:** 0.8981
- **Pipeline Time:** ~5 minutes

---

## ⚠️ Important Notes
- The `/data/` folder is empty — download CSV files from Kaggle separately
- Each teammate needs their own free Groq API key
- Never share or commit your API key to GitHub
- The `/models/` and `/reports/` folders are auto-generated when you run the project
```

---

After updating README.md, push it to GitHub:
```
git add .
git commit -m "Add README with setup instructions for teammates"
git push

## 🏗️ Architecture

```
You run: python main.py
              ↓
    ┌─────────────────────┐
    │  LANGGRAPH          │  ← Controls the flow
    │  ORCHESTRATOR       │
    └────────┬────────────┘
             │
   ┌─────────▼──────────┐
   │  Agent 1: EDA       │  → Loads, merges, analyzes data
   └─────────┬──────────┘
             │
   ┌─────────▼──────────┐
   │  Agent 2: Feature  │  → Cleans, encodes, engineers features
   └─────────┬──────────┘
             │
   ┌─────────▼──────────┐
   │  Agent 3: Model    │  → Applies SMOTE, trains XGBoost + LightGBM
   └─────────┬──────────┘
             │
   ┌─────────▼──────────┐
   │  Agent 4: Eval     │  → Checks AUC, loops back if too low
   └────┬────────────────┘
        │ (if AUC < 0.85 → retrain)
        │ (if AUC >= 0.85 → proceed)
   ┌────▼───────────────┐
   │  Agent 5: Decision │  → Predicts fraud + explains in English
   └────────────────────┘
             │
   ┌─────────▼──────────┐
   │  📄 Reports Saved  │  → /reports/ folder
   └────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| Agentic Framework | LangChain + LangGraph |
| LLM Brain | LLaMA 3 (70B) via Groq API (FREE) |
| ML Models | XGBoost + LightGBM |
| Imbalance Fix | SMOTE (imbalanced-learn) |
| Data Processing | Pandas + NumPy |
| Visualization | Matplotlib + Seaborn |

---

## ⚡ Quick Start

### Step 1 — Clone / Download the project
```bash
cd FraudSentinel_AI
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Get your FREE Groq API key
1. Go to https://console.groq.com
2. Sign up (free, no credit card needed)
3. Create an API key
4. Open `config.py` and paste your key:
```python
GROQ_API_KEY = "gsk_your_key_here"
```

### Step 4 — Download IEEE-CIS Dataset from Kaggle
1. Go to: https://www.kaggle.com/c/ieee-fraud-detection/data
2. Download these 2 files:
   - `train_transaction.csv`
   - `train_identity.csv`
3. Place both files in the `/data/` folder

### Step 5 — Run FraudSentinel AI
```bash
python main.py
```

That's it! The system runs everything autonomously.

---

## 📁 Project Structure

```
FraudSentinel_AI/
│
├── main.py                  ← Entry point (run this)
├── config.py                ← API keys + settings
├── requirements.txt         ← All dependencies
├── README.md                ← This file
│
├── data/
│   ├── train_transaction.csv   ← Download from Kaggle
│   └── train_identity.csv      ← Download from Kaggle
│
├── agents/
│   ├── eda_agent.py         ← Agent 1: Data Analysis
│   ├── feature_agent.py     ← Agent 2: Feature Engineering
│   ├── model_agent.py       ← Agent 3: Model Training
│   ├── eval_agent.py        ← Agent 4: Evaluation + Loop
│   └── decision_agent.py    ← Agent 5: Fraud Decision + Explanation
│
├── tools/
│   ├── data_tools.py        ← Data loading/cleaning functions
│   └── ml_tools.py          ← ML training/evaluation functions
│
├── orchestrator/
│   └── graph.py             ← LangGraph pipeline controller
│
├── models/
│   ├── xgboost_model.pkl    ← Saved after training
│   ├── lightgbm_model.pkl   ← Saved after training
│   └── best_model.pkl       ← Best model (used for predictions)
│
└── reports/
    ├── 01_eda_report.txt        ← EDA Agent findings
    ├── 02_feature_report.txt    ← Feature Engineering decisions
    ├── 03_model_report.txt      ← Training details
    ├── 04_eval_report.txt       ← Evaluation results
    └── 05_final_report.txt      ← Complete pipeline summary
```

---

## 🤖 What Each Agent Does

### Agent 1 — EDA Agent
- Loads and merges `train_transaction.csv` + `train_identity.csv`
- Detects missing values, class imbalance, column types
- Uses LLaMA 3 to reason about findings
- Output: cleaned data summary + recommendations

### Agent 2 — Feature Engineering Agent
- Drops columns with >50% missing values
- Creates 6 new features (log amount, night transaction flag, etc.)
- Label-encodes categorical columns
- Splits data into 80% train / 20% test

### Agent 3 — Model Training Agent
- Autonomously decides whether to apply SMOTE
- Trains XGBoost (200 trees, depth=6)
- Trains LightGBM (200 trees, depth=6)
- Saves both models to `/models/`

### Agent 4 — Evaluation Agent ⭐ (The Agentic Loop)
- Evaluates both models on AUC-ROC, F1, Precision, Recall
- Picks the best model
- **If AUC < 0.85 → loops back to Agent 3 to retrain**
- **If AUC >= 0.85 → passes to Agent 5**
- Maximum 3 retraining attempts

### Agent 5 — Decision Agent
- Takes new transactions and predicts fraud probability
- Uses LLaMA 3 to generate English explanations
- Example output: *"This transaction is flagged as HIGH RISK because: amount is 45x above card average, transaction occurred at 3AM, device seen for the first time..."*

---

## 📊 Expected Results

| Metric | Expected Value |
|---|---|
| AUC-ROC | 0.88 – 0.92 |
| Precision | 0.75 – 0.85 |
| Recall | 0.70 – 0.80 |
| F1 Score | 0.72 – 0.82 |

---

## 🎓 For Your College Viva

**One-line explanation:**
> "FraudSentinel AI is a LangGraph-based multi-agent system where five specialized AI agents collaboratively and autonomously detect financial fraud on the IEEE-CIS dataset, with a built-in retraining loop and natural language explanation of every fraud decision."

**Why it's novel compared to Kaggle solutions:**
1. **Autonomous decision-making** — no human decides which model, which features, or when to retrain
2. **Explainable AI** — every fraud decision comes with a plain-English reason
3. **Adaptive pipeline** — automatically retrains if performance is unsatisfactory
4. **Multi-agent architecture** — each agent specializes in one task (separation of concerns)

---

## 📝 Key Configuration (config.py)

```python
MIN_AUC_THRESHOLD = 0.85   # retrain if AUC below this
FRAUD_THRESHOLD   = 0.5    # flag as fraud if probability > 50%
MAX_MISSING_RATIO = 0.5    # drop column if >50% values missing
TEST_SIZE         = 0.2    # 80% train, 20% test
```

---

## 👤 Project Info

- **Project Name:** FraudSentinel AI
- **Dataset:** IEEE-CIS Fraud Detection (Kaggle)
- **Domain:** Agentic AI + Data Science
- **Framework:** LangChain + LangGraph
- **LLM:** LLaMA 3 70B via Groq

---


