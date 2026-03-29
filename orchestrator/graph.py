# ─────────────────────────────────────────────────────────────
#  FraudSentinel AI — orchestrator/graph.py
#
#  THE ORCHESTRATOR (Master Controller)
#
#  WHAT IS THIS?
#  LangGraph lets us define the FLOW between agents as a graph.
#  A graph has NODES (agents) and EDGES (connections between them).
#
#  FLOW:
#  EDA → Feature Engineering → Model Training → Evaluation
#  ↑                                                  |
#  └───────────── (if AUC too low, loop back) ────────┘
#                                    ↓ (if AUC is good)
#                             Decision Agent
#
#  The key feature: the conditional edge after Evaluation
#  decides whether to retrain or move to Decision Agent.
# ─────────────────────────────────────────────────────────────

from langgraph.graph import StateGraph, END
from typing import TypedDict, Any, Optional
import pandas as pd

from agents.eda_agent      import EDAAgent
from agents.feature_agent  import FeatureAgent
from agents.model_agent    import ModelAgent
from agents.eval_agent     import EvalAgent
from agents.decision_agent import DecisionAgent


# ══════════════════════════════════════════
#  STATE DEFINITION
#  This is the shared "notebook" all agents read from and write to.
#  TypedDict tells Python exactly what keys the state can have.
# ══════════════════════════════════════════
class FraudSentinelState(TypedDict, total=False):
    """
    The state dict passed between all agents in the pipeline.

    'total=False' means all keys are optional —
    early agents won't have keys that later agents add.
    """
    # From EDA Agent
    df                  : Any          # raw merged DataFrame
    columns_to_drop     : list         # high-missing columns to remove
    is_imbalanced       : bool         # whether SMOTE is needed
    fraud_ratio         : float        # % of fraud transactions
    numerical_cols      : list
    categorical_cols    : list
    eda_summary         : str
    eda_llm_analysis    : str

    # From Feature Agent
    df_processed        : Any          # clean + encoded DataFrame
    X_train             : Any          # training features
    X_test              : Any          # testing features
    y_train             : Any          # training labels
    y_test              : Any          # testing labels
    feature_names       : list         # list of all feature names
    new_features        : list         # newly engineered feature names
    encoders            : dict         # LabelEncoder objects
    feature_summary     : str
    feature_llm_analysis: str

    # From Model Agent
    xgb_model           : Any          # trained XGBoost model
    lgb_model           : Any          # trained LightGBM model
    trained_models      : dict         # dict of all trained models
    X_train_final       : Any          # post-SMOTE training data
    y_train_final       : Any
    model_strategy      : str
    training_summary    : str
    retrain_count       : int          # how many times we retrained

    # From Eval Agent
    best_model          : Any          # the winning model
    best_model_name     : str
    best_model_path     : str
    best_auc            : float
    xgb_results         : dict
    lgb_results         : dict
    eval_decision       : str          # 'pass' or 'retrain'
    eval_llm_analysis   : str

    # From Decision Agent
    final_report        : str
    sample_reports      : list

    # Control fields
    status              : str          # current pipeline status
    agent               : str          # which agent just ran
    error               : Optional[str]# error message if something failed


# ══════════════════════════════════════════
#  NODE FUNCTIONS
#  LangGraph needs regular functions, not class methods.
#  These are thin wrappers around our agent classes.
# ══════════════════════════════════════════

# Initialize agents once (not inside functions, to avoid re-init)
_eda_agent      = EDAAgent()
_feature_agent  = FeatureAgent()
_model_agent    = ModelAgent()
_eval_agent     = EvalAgent()
_decision_agent = DecisionAgent()


def run_eda(state: FraudSentinelState) -> FraudSentinelState:
    """Node 1: Run EDA Agent"""
    return _eda_agent.run(state)


def run_features(state: FraudSentinelState) -> FraudSentinelState:
    """Node 2: Run Feature Engineering Agent"""
    return _feature_agent.run(state)


def run_model(state: FraudSentinelState) -> FraudSentinelState:
    """Node 3: Run Model Training Agent"""
    return _model_agent.run(state)


def run_eval(state: FraudSentinelState) -> FraudSentinelState:
    """Node 4: Run Evaluation Agent"""
    return _eval_agent.run(state)


def run_decision(state: FraudSentinelState) -> FraudSentinelState:
    """Node 5: Run Decision Agent"""
    return _decision_agent.run(state)


# ══════════════════════════════════════════
#  CONDITIONAL EDGE
#  This is the KEY function that decides whether to:
#    - Loop BACK to retraining (if AUC is too low)
#    - Move FORWARD to Decision Agent (if AUC is good)
# ══════════════════════════════════════════
def should_retrain(state: FraudSentinelState) -> str:
    """
    After evaluation, decide next step.

    Returns:
      "retrain"  → go back to Model Agent
      "proceed"  → go to Decision Agent

    This conditional edge is what makes the system ADAPTIVE.
    It's not a fixed pipeline — it can loop back!
    """
    decision = state.get("eval_decision", "pass")
    if decision == "retrain":
        print(f"\n  🔄 ORCHESTRATOR: Sending back to Model Agent for retraining...")
        return "retrain"
    else:
        print(f"\n  ✅ ORCHESTRATOR: Evaluation passed. Moving to Decision Agent...")
        return "proceed"


# ══════════════════════════════════════════
#  BUILD THE LANGGRAPH
#  This assembles all nodes and edges into the final graph.
# ══════════════════════════════════════════
def build_graph():
    """
    Builds and compiles the FraudSentinel AI LangGraph.

    WHAT IS A GRAPH?
    Think of it like a flowchart:
      - add_node()  → adds an agent as a step
      - add_edge()  → connects two agents in sequence
      - add_conditional_edges() → connects based on a decision

    The graph is compiled once and can be run with .invoke()
    """
    # Create the graph with our state definition
    graph = StateGraph(FraudSentinelState)

    # ── ADD NODES (each agent is a node) ──
    graph.add_node("eda_agent"     , run_eda)
    graph.add_node("feature_agent" , run_features)
    graph.add_node("model_agent"   , run_model)
    graph.add_node("eval_agent"    , run_eval)
    graph.add_node("decision_agent", run_decision)

    # ── ADD EDGES (define the flow) ──

    # Fixed edges (always go in this direction)
    graph.add_edge("eda_agent"    , "feature_agent")
    graph.add_edge("feature_agent", "model_agent")
    graph.add_edge("model_agent"  , "eval_agent")

    # Conditional edge: after eval, loop back OR proceed
    graph.add_conditional_edges(
        source   = "eval_agent",          # from eval agent
        path     = should_retrain,        # call this function to decide
        path_map = {                       # map return values to nodes
            "retrain": "model_agent",     # loop back to retrain
            "proceed": "decision_agent"   # move forward
        }
    )

    # Final edge: decision agent → END
    graph.add_edge("decision_agent", END)

    # ── SET ENTRY POINT ──
    graph.set_entry_point("eda_agent")

    # ── COMPILE ──
    compiled = graph.compile()
    return compiled


# ══════════════════════════════════════════
#  MAIN PIPELINE RUNNER
# ══════════════════════════════════════════
def run_pipeline() -> dict:
    """
    Runs the complete FraudSentinel AI pipeline.

    This is called from main.py with just one line:
        result = run_pipeline()

    The graph handles everything else automatically.
    """
    print(f"\n{'='*70}")
    print(f"  🛡️  FRAUDSENTINEL AI — PIPELINE STARTING")
    print(f"{'='*70}\n")

    # Build the graph
    graph = build_graph()

    # Start with empty state — agents will fill it in as they run
    initial_state = {
        "status"       : "starting",
        "retrain_count": 0
    }

    # Run the full pipeline
    final_state = graph.invoke(initial_state)

    return final_state
