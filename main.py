"""
Evo-Hierarch-RAG — Entry Point.

Usage
-----
    python main.py
    python main.py --query "What are the latest AI developments?"

Environment variables (see .env.example):
    OPENAI_API_KEY   — required
    TAVILY_API_KEY   — required for Research Agent web search
    LLM_MODEL        — default: gpt-4o-mini
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

# Configure logging before importing project modules so all loggers pick it up.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def run_query(query: str) -> Dict[str, Any]:
    """Execute a single query through the full Hierarchical RAG pipeline."""
    from src.config import MAX_RETRIES
    from src.graph import graph

    initial_state: Dict[str, Any] = {
        "original_query": query,
        "current_query": query,
        "subtasks": [],
        "routing_decision": [],
        "agent_results": [],
        "aggregated_context": "",
        "relevancy_score": 0.0,
        "hallucination_detected": False,
        "evaluation_feedback": "",
        "retry_count": 0,
        "max_retries": MAX_RETRIES,
        "failed_cases": [],
        "system_prompts": {},
        "few_shot_examples": [],
        "final_answer": None,
    }

    logger.info("=== Starting pipeline for query: %s ===", query)
    final_state: Dict[str, Any] = graph.invoke(initial_state)
    return final_state


def print_result(state: Dict[str, Any]) -> None:
    """Pretty-print the final pipeline result."""
    divider = "─" * 70
    print(f"\n{divider}")
    print("QUERY :", state.get("original_query", ""))
    print(divider)
    print("ROUTING:", ", ".join(state.get("routing_decision", [])))
    print(divider)

    for result in state.get("agent_results", []):
        print(f"\n[{result['agent_name'].upper()}]")
        print(f"  Sub-query : {result['sub_query']}")
        snippet = result['result'][:300].replace("\n", " ")
        print(f"  Result    : {snippet}…" if len(result['result']) > 300 else f"  Result    : {result['result']}")

    print(f"\n{divider}")
    print("EVALUATION")
    print(f"  Relevancy score     : {state.get('relevancy_score', 'N/A'):.2f}")
    print(f"  Hallucination found : {state.get('hallucination_detected', 'N/A')}")
    print(f"  Feedback            : {state.get('evaluation_feedback', '')}")
    print(f"  Retry count         : {state.get('retry_count', 0)}")
    print(f"\n{divider}")
    print("FINAL ANSWER")
    print(state.get("final_answer", "(no answer generated)"))
    print(divider)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evo-Hierarch-RAG demo")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query to run (omit for interactive demo)",
    )
    args = parser.parse_args()

    demo_queries = [
        "大型语言模型研究的最新突破是什么？",
        # "我们有什么库存产品，价格是多少？",
        # "计算所有订单的总收入，并按支出显示顶级客户。",
    ]

    queries_to_run = [args.query] if args.query else demo_queries

    for q in queries_to_run:
        try:
            result = run_query(q)
            print_result(result)
        except Exception as exc:  # noqa: BLE001
            logger.error("Pipeline failed for query '%s': %s", q, exc, exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
