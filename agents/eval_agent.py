# FraudSentinel AI — agents/eval_agent.py

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from tools.ml_tools import evaluate_model
from tools.data_tools import save_report
from config import GROQ_API_KEY, LLM_MODEL, REPORTS_DIR, MIN_AUC_THRESHOLD
import joblib
import os


class EvalAgent:

    def __init__(self):
        self.llm = ChatGroq(
            api_key    = GROQ_API_KEY,
            model_name = LLM_MODEL,
            temperature= 0
        )
        self.agent_name      = "Evaluation Agent"
        self.max_retrain_attempts = 3

    def _analyze_short(self, xgb_results: dict, lgb_results: dict) -> str:
        """Short version — shown in terminal."""
        system_prompt = """You are the Evaluation Agent in FraudSentinel AI.
Respond in exactly 3 bullet points. Each bullet = one sentence, max 15 words.
Start each bullet with a dash (-). No headers, no paragraphs."""

        user_message = f"""XGBoost: AUC={xgb_results['auc']:.4f}, F1={xgb_results['f1']:.4f}, Recall={xgb_results['recall']:.4f}
LightGBM: AUC={lgb_results['auc']:.4f}, F1={lgb_results['f1']:.4f}, Recall={lgb_results['recall']:.4f}
Threshold: {MIN_AUC_THRESHOLD}

Give exactly 3 bullets:
- Which model is better and why
- Is this performance acceptable for fraud detection
- PASS or RETRAIN recommendation and main reason"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        return self.llm.invoke(messages).content

    def _analyze_detailed(self, xgb_results: dict, lgb_results: dict) -> str:
        """Detailed version — saved in report."""
        system_prompt = """You are an expert Model Evaluation agent in FraudSentinel AI.
You evaluate ML models for fraud detection and make pass/fail decisions.
Provide detailed, thorough analysis. Be strict — fraud detection models must be reliable."""

        user_message = f"""I have evaluated two fraud detection models. Here are the full results:

XGBoost Results:
  AUC-ROC   : {xgb_results['auc']:.4f}
  F1 Score  : {xgb_results['f1']:.4f}
  Precision : {xgb_results['precision']:.4f}
  Recall    : {xgb_results['recall']:.4f}

LightGBM Results:
  AUC-ROC   : {lgb_results['auc']:.4f}
  F1 Score  : {lgb_results['f1']:.4f}
  Precision : {lgb_results['precision']:.4f}
  Recall    : {lgb_results['recall']:.4f}

Minimum acceptable AUC threshold: {MIN_AUC_THRESHOLD}

Please provide detailed analysis covering:
1. Which model is better overall, and explain why in detail
2. Is the performance acceptable for a production fraud detection system?
3. Discuss the tradeoffs between Precision and Recall for this specific use case
4. If retraining were needed, what specific changes would you recommend?
5. Final recommendation: PASS or RETRAIN, with full justification"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        return self.llm.invoke(messages).content

    def run(self, state: dict) -> dict:
        print(f"\n{'='*60}")
        print(f"  Evaluation Agent — STARTING")
        print(f"{'='*60}")

        xgb_model     = state["xgb_model"]
        lgb_model     = state["lgb_model"]
        X_test        = state["X_test"]
        y_test        = state["y_test"]
        retrain_count = state.get("retrain_count", 0)
        all_summaries = []

        print("\n[Step 1/3] Evaluating XGBoost...")
        xgb_results = evaluate_model(xgb_model, "XGBoost", X_test, y_test)
        all_summaries.append("=== XGBOOST EVALUATION ===\n" + xgb_results["summary"])
        print(f"  XGBoost AUC: {xgb_results['auc']:.4f}")

        print("\n[Step 2/3] Evaluating LightGBM...")
        lgb_results = evaluate_model(lgb_model, "LightGBM", X_test, y_test)
        all_summaries.append("=== LIGHTGBM EVALUATION ===\n" + lgb_results["summary"])
        print(f"  LightGBM AUC: {lgb_results['auc']:.4f}")

        if xgb_results["auc"] >= lgb_results["auc"]:
            best_model      = xgb_model
            best_model_name = "XGBoost"
            best_results    = xgb_results
        else:
            best_model      = lgb_model
            best_model_name = "LightGBM"
            best_results    = lgb_results

        print(f"\n  Best model: {best_model_name} (AUC: {best_results['auc']:.4f})")

        # SHORT → terminal
        print("\n[LLM] Evaluation Agent analyzing results...")
        short_analysis = self._analyze_short(xgb_results, lgb_results)
        print("\n  LLM Decision:")
        for line in [l.strip() for l in short_analysis.split("\n") if l.strip()]:
            print(f"     {line}")

        # DETAILED → report
        detailed_analysis = self._analyze_detailed(xgb_results, lgb_results)

        best_auc = best_results["auc"]
        if best_auc >= MIN_AUC_THRESHOLD:
            decision    = "pass"
            next_status = "evaluation_passed"
            print(f"\n  DECISION: PASS (AUC {best_auc:.4f} >= threshold {MIN_AUC_THRESHOLD})")
        elif retrain_count >= self.max_retrain_attempts:
            decision    = "pass"
            next_status = "evaluation_passed"
            print(f"\n  DECISION: FORCED PASS (max retrains reached, best AUC: {best_auc:.4f})")
        else:
            decision    = "retrain"
            next_status = "needs_retraining"
            print(f"\n  DECISION: RETRAIN (AUC {best_auc:.4f} < threshold {MIN_AUC_THRESHOLD})")

        best_model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "models", "best_model.pkl"
        )
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        joblib.dump(best_model, best_model_path)

        combined_summary = "\n\n".join(all_summaries)
        decision_summary = (
            f"\n\n=== FINAL DECISION ===\n"
            f"Best Model   : {best_model_name}\n"
            f"Best AUC     : {best_auc:.4f}\n"
            f"Threshold    : {MIN_AUC_THRESHOLD}\n"
            f"Decision     : {decision.upper()}\n"
            f"Retrain Count: {retrain_count}"
        )
        full_report = (
            f"FRAUDSENTINEL AI — EVALUATION AGENT REPORT\n"
            f"{'='*50}\n\n"
            f"{combined_summary}\n\n"
            f"{'='*50}\n"
            f"LLM REASONING:\n{detailed_analysis}"
            f"{decision_summary}"
        )
        report_path = save_report("04_eval_report", full_report, REPORTS_DIR)
        print(f"  Report saved: {report_path}")
        print(f"\n  Evaluation Agent COMPLETE — Decision: {decision.upper()}")

        return {
            **state,
            "best_model"       : best_model,
            "best_model_name"  : best_model_name,
            "best_model_path"  : best_model_path,
            "best_auc"         : best_auc,
            "xgb_results"      : xgb_results,
            "lgb_results"      : lgb_results,
            "eval_decision"    : decision,
            "eval_llm_analysis": detailed_analysis,
            "retrain_count"    : retrain_count + (1 if decision == "retrain" else 0),
            "status"           : next_status,
            "agent"            : self.agent_name
        }