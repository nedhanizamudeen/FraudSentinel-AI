# FraudSentinel AI — agents/feature_agent.py

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from tools.ml_tools import (
    clean_data, encode_categoricals,
    engineer_features, split_data
)
from tools.data_tools import save_report
from config import GROQ_API_KEY, LLM_MODEL, REPORTS_DIR


class FeatureAgent:

    def __init__(self):
        self.llm = ChatGroq(
            api_key    = GROQ_API_KEY,
            model_name = LLM_MODEL,
            temperature= 0
        )
        self.agent_name = "Feature Engineering Agent"

    def _think_short(self, feature_summary: str, eda_analysis: str) -> str:
        """Short version — shown in terminal."""
        system_prompt = """You are the Feature Engineering Agent in FraudSentinel AI.
Respond in exactly 4 bullet points. Each bullet = one sentence, max 15 words.
Start each bullet with a dash (-). No headers, no paragraphs."""

        user_message = f"""EDA finding (brief): {eda_analysis[:200]}

Feature steps done: {feature_summary[:300]}

Give exactly 4 bullets:
- Overall quality of the feature engineering
- Most useful new feature for fraud detection
- Any risk or concern
- One additional recommendation"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        return self.llm.invoke(messages).content

    def _think_detailed(self, feature_summary: str, eda_analysis: str) -> str:
        """Detailed version — saved in report."""
        system_prompt = """You are an expert Feature Engineering agent in FraudSentinel AI.
Your job is to make intelligent, well-reasoned decisions about feature engineering for fraud detection.
Be thorough, specific, and explain the reasoning behind each decision in detail."""

        user_message = f"""The EDA Agent found the following about our data:
{eda_analysis}

I have now performed the following feature engineering steps:
{feature_summary}

Please provide a detailed analysis covering:
1. Evaluate whether the feature engineering steps are appropriate for fraud detection and why
2. Explain in detail which new features are most likely to help detect fraud and why
3. Identify any potential risks or issues with the current approach
4. Recommend any additional features that should be created, with reasoning"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        return self.llm.invoke(messages).content

    def run(self, state: dict) -> dict:
        print(f"\n{'='*60}")
        print(f"  Feature Engineering Agent — STARTING")
        print(f"{'='*60}")

        df              = state["df"]
        columns_to_drop = state["columns_to_drop"]
        eda_analysis    = state.get("eda_llm_analysis", "")
        all_summaries   = []

        print("\n[Step 1/4] Cleaning data...")
        clean_result = clean_data(df, columns_to_drop)
        df_clean     = clean_result["data"]
        all_summaries.append("=== DATA CLEANING ===\n" + clean_result["summary"])
        print(f"  {clean_result['summary']}")

        print("\n[Step 2/4] Engineering new features...")
        feature_result = engineer_features(df_clean)
        df_featured    = feature_result["data"]
        all_summaries.append("=== FEATURE ENGINEERING ===\n" + feature_result["summary"])
        print(f"  {feature_result['summary']}")

        print("\n[Step 3/4] Encoding categorical columns...")
        encode_result = encode_categoricals(df_featured)
        df_encoded    = encode_result["data"]
        all_summaries.append("=== ENCODING ===\n" + encode_result["summary"])
        print(f"  {encode_result['summary']}")

        print("\n[Step 4/4] Splitting data into train/test sets...")
        split_result = split_data(df_encoded)
        all_summaries.append("=== DATA SPLIT ===\n" + split_result["summary"])
        print(f"  {split_result['summary']}")

        combined_summary = "\n\n".join(all_summaries)

        # SHORT → terminal
        print("\n[LLM] Feature Agent is reasoning about decisions...")
        short_analysis = self._think_short(combined_summary, eda_analysis)
        print("\n  LLM Decision:")
        for line in [l.strip() for l in short_analysis.split("\n") if l.strip()]:
            print(f"     {line}")

        # DETAILED → report
        detailed_analysis = self._think_detailed(combined_summary, eda_analysis)

        full_report = (
            f"FRAUDSENTINEL AI — FEATURE ENGINEERING AGENT REPORT\n"
            f"{'='*50}\n\n"
            f"{combined_summary}\n\n"
            f"{'='*50}\n"
            f"LLM REASONING:\n{detailed_analysis}"
        )
        report_path = save_report("02_feature_report", full_report, REPORTS_DIR)
        print(f"  Report saved: {report_path}")
        print(f"\n  Feature Engineering Agent COMPLETE")

        return {
            **state,
            "df_processed"        : df_encoded,
            "X_train"             : split_result["X_train"],
            "X_test"              : split_result["X_test"],
            "y_train"             : split_result["y_train"],
            "y_test"              : split_result["y_test"],
            "feature_names"       : split_result["feature_names"],
            "new_features"        : feature_result["new_features"],
            "encoders"            : encode_result["encoders"],
            "feature_summary"     : combined_summary,
            "feature_llm_analysis": detailed_analysis,
            "status"              : "features_ready",
            "agent"               : self.agent_name
        }