# ─────────────────────────────────────────────────────────────
#  FraudSentinel AI — main.py
#  Run with: python main.py
# ─────────────────────────────────────────────────────────────

import sys
import os
import time


def print_banner():
    print("\n" + "="*70)
    print("  FraudSentinel AI  ")
    print("  Autonomous Multi-Agent Fraud Detection System")
    print("  Dataset  : IEEE-CIS Financial Transaction Data")
    print("  LLM      : LLaMA 3 via Groq")
    print("  Framework: LangChain + LangGraph")
    print("  Agents   : EDA -> Feature -> Model -> Eval -> Decision")
    print("="*70)


def check_setup() -> bool:
    from config import GROQ_API_KEY, TRANSACTION_FILE, IDENTITY_FILE

    all_good = True

    if GROQ_API_KEY == "paste_your_groq_key_here":
        print("  ❌ GROQ_API_KEY not set!")
        print("     → Get your free key at: https://console.groq.com")
        print("     → Paste it in config.py")
        all_good = False
    else:
        print("  ✓ Groq API key found")

    if not os.path.exists(TRANSACTION_FILE):
        print("  ❌ train_transaction.csv not found in /data/")
        print("     → Download from: https://www.kaggle.com/c/ieee-fraud-detection")
        all_good = False
    else:
        size_mb = os.path.getsize(TRANSACTION_FILE) / 1e6
        print(f"  ✓ train_transaction.csv found ({size_mb:.0f} MB)")

    if not os.path.exists(IDENTITY_FILE):
        print("  ❌ train_identity.csv not found in /data/")
        all_good = False
    else:
        size_mb = os.path.getsize(IDENTITY_FILE) / 1e6
        print(f"  ✓ train_identity.csv found ({size_mb:.0f} MB)")

    return all_good


def run_full_pipeline():
    from orchestrator.graph import run_pipeline

    print("\n  Starting full pipeline...\n")
    start_time = time.time()

    final_state = run_pipeline()

    elapsed = time.time() - start_time
    mins    = int(elapsed // 60)
    secs    = int(elapsed % 60)

    print(f"\n  Pipeline completed in {mins}m {secs}s")

    if "best_auc" in final_state:
        print(f"  Best Model : {final_state.get('best_model_name', 'N/A')}")
        print(f"  Best AUC   : {final_state.get('best_auc', 0):.4f}")

    return final_state


def main():
    print_banner()

    mode = "--pipeline"
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    print(f"\n  Mode: {mode}")
    print(f"\n  Checking setup...")

    if not check_setup():
        print("\n  Setup incomplete. Please fix the issues above and try again.\n")
        sys.exit(1)

    print("\n  All checks passed!\n")

    if mode == "--demo":
        run_demo()
    elif mode == "--pipeline" or mode is None:
        run_full_pipeline()
    else:
        print(f"  Unknown mode: {mode}")
        print(f"  Usage: python main.py [--pipeline|--demo]")


if __name__ == "__main__":
    main()