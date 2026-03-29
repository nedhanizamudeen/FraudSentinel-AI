# FraudSentinel AI — agents/decision_agent.py

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from tools.ml_tools import predict_transaction
from tools.data_tools import save_report
from config import GROQ_API_KEY, LLM_MODEL, REPORTS_DIR, FRAUD_THRESHOLD
import json
import os
from datetime import datetime


class DecisionAgent:

    def __init__(self):
        self.llm = ChatGroq(
            api_key    = GROQ_API_KEY,
            model_name = LLM_MODEL,
            temperature= 0.1
        )
        self.agent_name = "Decision Agent"

    def _explain_short(self, transaction: dict, fraud_probability: float,
                       risk_level: str, is_fraud: bool, model_name: str) -> str:
        """Short explanation — shown in terminal."""
        system_prompt = """You are FraudSentinel AI's Decision Agent.
Respond in exactly 4 bullet points. Each bullet = one sentence, max 15 words.
Start each bullet with a dash (-). No headers, no paragraphs."""

        key_features = {k: v for k, v in list(transaction.items())[:12]}
        user_message = f"""Transaction: {json.dumps(key_features, default=str)}
Model: {model_name} | Probability: {fraud_probability*100:.1f}% | Risk: {risk_level}
Decision: {'FRAUD' if is_fraud else 'LEGITIMATE'}

Give exactly 4 bullets:
- One-sentence verdict
- Main suspicious or safe feature observed
- Recommended action
- Confidence level"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        return self.llm.invoke(messages).content

    def _explain_detailed(self, transaction: dict, fraud_probability: float,
                          risk_level: str, is_fraud: bool, model_name: str) -> str:
        """Detailed explanation — saved in report."""
        system_prompt = """You are FraudSentinel AI's Decision Agent — an expert fraud analyst.
You receive transaction details and fraud probability scores from ML models.
Provide a thorough, detailed explanation of the fraud decision.

Be specific about which features are suspicious and why.
Write as if you are briefing a fraud analyst at a bank with full detail."""

        user_message = f"""Transaction Analysis Request:

Transaction Details:
{json.dumps(transaction, indent=2, default=str)}

Model Assessment:
  - Model Used       : {model_name}
  - Fraud Probability: {fraud_probability*100:.1f}%
  - Risk Level       : {risk_level}
  - Decision         : {'FRAUD DETECTED' if is_fraud else 'LEGITIMATE'}

Please provide a full detailed analysis covering:
1. A clear verdict statement with reasoning
2. All key risk factors (or safety factors) observed in this transaction with explanations
3. Which specific feature values are most suspicious or safe, and why
4. Recommended action for the fraud team with justification
5. Confidence assessment (high/medium/low) and detailed reasoning"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        return self.llm.invoke(messages).content

    def analyze_transaction(self, transaction: dict, best_model,
                            feature_names: list, model_name: str = "Best Model") -> dict:

        pred_result    = predict_transaction(best_model, transaction, feature_names)
        fraud_prob     = pred_result["fraud_probability"]
        is_fraud       = pred_result["is_fraud"]
        risk_level     = pred_result["risk_level"]

        # SHORT for terminal display
        short_expl = self._explain_short(transaction, fraud_prob,
                                         risk_level, is_fraud, model_name)
        # DETAILED for report
        detailed_expl = self._explain_detailed(transaction, fraud_prob,
                                                risk_level, is_fraud, model_name)

        verdict_text = "FRAUD DETECTED" if is_fraud else "LEGITIMATE"

        # Terminal report uses short explanation
        report = f"""
{'='*60}
  FRAUDSENTINEL AI — TRANSACTION VERDICT
{'='*60}
  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Model     : {model_name}

  VERDICT: {verdict_text}

  Fraud Probability : {fraud_prob*100:.1f}%
  Risk Level        : {risk_level}
  Threshold Used    : {FRAUD_THRESHOLD*100:.0f}%

{'='*60}
  LLM EXPLANATION:
{'='*60}
{short_expl}
{'='*60}
"""
        return {
            "fraud_probability": fraud_prob,
            "is_fraud"         : is_fraud,
            "risk_level"       : risk_level,
            "verdict"          : verdict_text,
            "short_explanation": short_expl,
            "detailed_explanation": detailed_expl,
            "report"           : report
        }

    def run(self, state: dict) -> dict:
        print(f"\n{'='*60}")
        print(f"  Decision Agent — STARTING")
        print(f"{'='*60}")

        best_model      = state["best_model"]
        best_model_name = state["best_model_name"]
        best_auc        = state["best_auc"]
        X_test          = state["X_test"]
        y_test          = state["y_test"]
        feature_names   = state["feature_names"]

        print("\n[Step 1/2] Analyzing sample fraud transactions...")

        fraud_indices  = y_test[y_test == 1].index[:3]
        sample_reports = []
        detailed_reports = []

        for i, idx in enumerate(fraud_indices):
            print(f"\n  Analyzing fraud transaction {i+1}/3...")
            transaction = X_test.loc[idx].to_dict()
            result = self.analyze_transaction(
                transaction   = transaction,
                best_model    = best_model,
                feature_names = feature_names,
                model_name    = best_model_name
            )
            sample_reports.append(result["report"])
            detailed_reports.append(
                f"--- Transaction {i+1} ---\n"
                f"Fraud Probability: {result['fraud_probability']*100:.1f}% | "
                f"Risk: {result['risk_level']} | Verdict: {result['verdict']}\n\n"
                f"DETAILED ANALYSIS:\n{result['detailed_explanation']}\n"
            )
            print(f"  -> Predicted: {'FRAUD' if result['is_fraud'] else 'LEGITIMATE'} "
                  f"(Actual: FRAUD) | Prob: {result['fraud_probability']*100:.1f}%")

        print("\n[Step 2/2] Generating final pipeline report...")

        # Terminal output — compact
        final_report = f"""
{'='*70}
  FRAUDSENTINEL AI — FINAL PIPELINE REPORT
{'='*70}
  Project    : FraudSentinel AI v1.0.0
  Dataset    : IEEE-CIS Fraud Detection
  Completed  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

  PIPELINE RESULTS:
  -----------------------------------------
  Best Model     : {best_model_name}
  Best AUC-ROC   : {best_auc:.4f}
  XGBoost AUC    : {state['xgb_results']['auc']:.4f}
  LightGBM AUC   : {state['lgb_results']['auc']:.4f}
  Retrain Loops  : {state.get('retrain_count', 0)}
  Features Used  : {len(feature_names)}

  SAMPLE FRAUD TRANSACTION ANALYSES:
  -----------------------------------------
{''.join(sample_reports)}
{'='*70}
  Pipeline completed autonomously by FraudSentinel AI.
  All reports saved to: /reports/
{'='*70}
"""
        print(final_report)

        # Full report file — uses detailed explanations
        full_report_file = f"""FRAUDSENTINEL AI — FINAL PIPELINE REPORT
{'='*70}
Project    : FraudSentinel AI v1.0.0
Dataset    : IEEE-CIS Fraud Detection
Completed  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PIPELINE RESULTS:
{'='*50}
Best Model     : {best_model_name}
Best AUC-ROC   : {best_auc:.4f}
XGBoost AUC    : {state['xgb_results']['auc']:.4f}
LightGBM AUC   : {state['lgb_results']['auc']:.4f}
Retrain Loops  : {state.get('retrain_count', 0)}
Features Used  : {len(feature_names)}

DETAILED FRAUD TRANSACTION ANALYSES:
{'='*50}
{chr(10).join(detailed_reports)}
{'='*70}
Pipeline completed autonomously by FraudSentinel AI.
"""
        save_report("05_final_report", full_report_file, REPORTS_DIR)

        print(f"\n  Decision Agent COMPLETE")
        print(f"\n{'='*70}")
        print(f"  FRAUDSENTINEL AI PIPELINE COMPLETE!")
        print(f"  Best Model : {best_model_name} | AUC: {best_auc:.4f}")
        print(f"  All reports saved in /reports/ folder")
        print(f"{'='*70}\n")

        return {
            **state,
            "final_report"  : final_report,
            "sample_reports": sample_reports,
            "status"        : "pipeline_complete",
            "agent"         : self.agent_name
        }