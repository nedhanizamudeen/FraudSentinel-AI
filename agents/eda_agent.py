# FraudSentinel AI — agents/eda_agent.py

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from tools.data_tools import (
    load_and_merge_data,
    analyze_missing_values,
    analyze_class_balance,
    get_column_types,
    save_report
)
from config import GROQ_API_KEY, LLM_MODEL, REPORTS_DIR


class EDAAgent:

    def __init__(self):
        self.llm = ChatGroq(
            api_key    = GROQ_API_KEY,
            model_name = LLM_MODEL,
            temperature= 0
        )
        self.agent_name = "EDA Agent"

    def _think_short(self, data_summary: str) -> str:
        """Short version — shown in terminal (4 bullets)."""
        system_prompt = """You are the EDA Agent in FraudSentinel AI.
Respond in exactly 4 bullet points. Each bullet = one sentence, max 15 words.
Start each bullet with a dash (-). No headers, no paragraphs."""

        user_message = f"""Dataset stats:
{data_summary}

Give exactly 4 bullets:
- Most important finding about the data
- Main problem that needs fixing
- One key action for the Feature Engineering Agent
- Difficulty level of this dataset"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        return self.llm.invoke(messages).content

    def _think_detailed(self, data_summary: str) -> str:
        """Detailed version — saved in report only."""
        system_prompt = """You are an expert data scientist agent called EDA Agent,
working on the FraudSentinel AI project for IEEE-CIS fraud detection.

Your job is to:
1. Analyze the data statistics provided
2. Identify the most important problems or patterns
3. Make specific recommendations for the next steps
4. Explain your reasoning clearly and in detail

Be thorough, specific, and actionable. This analysis will be saved in a report."""

        user_message = f"""I have analyzed the IEEE-CIS Fraud Detection dataset.
Here are the complete findings:

{data_summary}

Please provide a detailed analysis covering:
1. Summary of the 3 most important findings (with explanations)
2. What problems need to be fixed and exactly why each is a problem
3. Specific recommended actions for the Feature Engineering Agent
4. Assessment of this dataset's difficulty level and reasoning"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        return self.llm.invoke(messages).content

    def run(self, state: dict) -> dict:
        print(f"\n{'='*60}")
        print(f"  EDA Agent — STARTING")
        print(f"{'='*60}")

        print("\n[Step 1/4] Loading IEEE-CIS dataset...")
        load_result = load_and_merge_data()

        if load_result["status"] == "error":
            print(f"  ERROR: {load_result['summary']}")
            return {**state, "error": load_result["summary"],
                    "status": "failed", "agent": self.agent_name}

        df = load_result["data"]
        all_summaries = ["=== DATA LOADING ===\n" + load_result["summary"]]

        print("\n[Step 2/4] Analyzing missing values...")
        missing_result = analyze_missing_values(df)
        all_summaries.append("=== MISSING VALUES ===\n" + missing_result["summary"])
        print(f"  Found {len(missing_result['columns_to_drop'])} columns to drop")

        print("\n[Step 3/4] Checking class balance...")
        balance_result = analyze_class_balance(df)
        all_summaries.append("=== CLASS BALANCE ===\n" + balance_result["summary"])

        print("\n[Step 4/4] Identifying column types...")
        type_result = get_column_types(df)
        all_summaries.append("=== COLUMN TYPES ===\n" + type_result["summary"])

        combined_summary = "\n\n".join(all_summaries)

        # SHORT version → printed in terminal
        print("\n[LLM] EDA Agent is reasoning about findings...")
        short_analysis = self._think_short(combined_summary)
        print("\n  LLM Decision:")
        for line in [l.strip() for l in short_analysis.split("\n") if l.strip()]:
            print(f"     {line}")

        # DETAILED version → saved in report
        detailed_analysis = self._think_detailed(combined_summary)

        full_report = (
            f"FRAUDSENTINEL AI — EDA AGENT REPORT\n"
            f"{'='*50}\n\n"
            f"{combined_summary}\n\n"
            f"{'='*50}\n"
            f"LLM REASONING:\n{detailed_analysis}"
        )
        report_path = save_report("01_eda_report", full_report, REPORTS_DIR)
        print(f"  Report saved: {report_path}")
        print(f"\n  EDA Agent COMPLETE")

        return {
            **state,
            "df"               : df,
            "columns_to_drop"  : missing_result["columns_to_drop"],
            "is_imbalanced"    : balance_result["is_imbalanced"],
            "fraud_ratio"      : balance_result["fraud_ratio"],
            "numerical_cols"   : type_result["numerical_cols"],
            "categorical_cols" : type_result["categorical_cols"],
            "eda_summary"      : combined_summary,
            "eda_llm_analysis" : detailed_analysis,
            "status"           : "eda_complete",
            "agent"            : self.agent_name
        }