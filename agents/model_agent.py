# FraudSentinel AI — agents/model_agent.py

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from tools.ml_tools import apply_smote, train_xgboost, train_lightgbm
from tools.data_tools import save_report
from config import GROQ_API_KEY, LLM_MODEL, REPORTS_DIR


class ModelAgent:

    def __init__(self):
        self.llm = ChatGroq(
            api_key    = GROQ_API_KEY,
            model_name = LLM_MODEL,
            temperature= 0
        )
        self.agent_name = "Model Training Agent"

    def _decide_short(self, state: dict) -> str:
        """Short version — shown in terminal."""
        system_prompt = """You are the Model Training Agent in FraudSentinel AI.
Respond in exactly 3 bullet points. Each bullet = one sentence, max 15 words.
Start each bullet with a dash (-). No headers, no paragraphs."""

        user_message = f"""Dataset: imbalanced={state.get('is_imbalanced', True)}, fraud={state.get('fraud_ratio', 3.5):.1f}%, features={len(state.get('feature_names', []))}

Give exactly 3 bullets:
- Should we apply SMOTE and why
- Which model to train first and why
- One special consideration for this dataset"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        return self.llm.invoke(messages).content

    def _decide_detailed(self, state: dict) -> str:
        """Detailed version — saved in report."""
        system_prompt = """You are an expert ML Engineer agent in FraudSentinel AI.
You specialize in training fraud detection models on imbalanced financial datasets.
Provide detailed, well-reasoned decisions about the training strategy."""

        user_message = f"""Based on the EDA and Feature Engineering, here is what we know:

- Dataset imbalanced: {state.get('is_imbalanced', True)}
- Fraud ratio: {state.get('fraud_ratio', 3.5):.2f}%
- Training samples: {len(state.get('X_train', []))}
- Number of features: {len(state.get('feature_names', []))}
- New engineered features: {state.get('new_features', [])}

EDA Agent finding: {state.get('eda_llm_analysis', 'N/A')[:500]}

Please provide a detailed training strategy covering:
1. Should we apply SMOTE? Explain thoroughly why or why not.
2. Which model to train first (XGBoost or LightGBM) and detailed reasoning.
3. What specific hyperparameter considerations apply to this dataset?
4. Any risks or special considerations for training on this data?"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        return self.llm.invoke(messages).content

    def run(self, state: dict) -> dict:
        print(f"\n{'='*60}")
        print(f"  Model Training Agent — STARTING")
        print(f"{'='*60}")

        X_train       = state["X_train"]
        y_train       = state["y_train"]
        is_imbalanced = state.get("is_imbalanced", True)
        all_summaries = []

        # SHORT → terminal
        print("\n[LLM] Model Agent deciding training strategy...")
        short_strategy = self._decide_short(state)
        print("\n  LLM Decision:")
        for line in [l.strip() for l in short_strategy.split("\n") if l.strip()]:
            print(f"     {line}")

        # DETAILED → report
        detailed_strategy = self._decide_detailed(state)

        X_train_final = X_train
        y_train_final = y_train

        if is_imbalanced:
            print("\n[Step 1/3] Applying SMOTE (dataset is imbalanced)...")
            smote_result  = apply_smote(X_train, y_train)
            X_train_final = smote_result["X_train_resampled"]
            y_train_final = smote_result["y_train_resampled"]
            all_summaries.append("=== SMOTE ===\n" + smote_result["summary"])
            print("  SMOTE applied successfully")
        else:
            print("\n[Step 1/3] Skipping SMOTE (dataset is balanced)")
            all_summaries.append("=== SMOTE ===\nSMOTE skipped — dataset is already balanced.")

        print("\n[Step 2/3] Training XGBoost...")
        xgb_result = train_xgboost(X_train_final, y_train_final)
        all_summaries.append("=== XGBOOST TRAINING ===\n" + xgb_result["summary"])
        print("  XGBoost trained and saved")

        print("\n[Step 3/3] Training LightGBM...")
        lgb_result = train_lightgbm(X_train_final, y_train_final)
        all_summaries.append("=== LIGHTGBM TRAINING ===\n" + lgb_result["summary"])
        print("  LightGBM trained and saved")

        combined_summary = "\n\n".join(all_summaries)
        full_report = (
            f"FRAUDSENTINEL AI — MODEL TRAINING AGENT REPORT\n"
            f"{'='*50}\n\n"
            f"=== LLM STRATEGY DECISION ===\n{detailed_strategy}\n\n"
            f"{combined_summary}"
        )
        report_path = save_report("03_model_report", full_report, REPORTS_DIR)
        print(f"  Report saved: {report_path}")
        print(f"\n  Model Training Agent COMPLETE")

        return {
            **state,
            "xgb_model"        : xgb_result["model"],
            "lgb_model"        : lgb_result["model"],
            "trained_models"   : {"XGBoost": xgb_result["model"], "LightGBM": lgb_result["model"]},
            "X_train_final"    : X_train_final,
            "y_train_final"    : y_train_final,
            "model_strategy"   : detailed_strategy,
            "training_summary" : combined_summary,
            "status"           : "models_trained",
            "agent"            : self.agent_name,
            "retrain_count"    : state.get("retrain_count", 0)
        }