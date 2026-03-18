"""
Query Rewriter Node — Self-Improving Component #2.

Triggered when the Evaluator determines that the retrieved context is not
relevant enough.  Rewrites ``current_query`` to improve retrieval quality
on the next iteration, and increments ``retry_count``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import LLM_MODEL, LLM_TEMPERATURE
from src.state import GraphState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_REWRITER_SYSTEM = """You are a Query Rewriting specialist for a RAG system.

The original query failed to retrieve sufficiently relevant context.

Given:
- The original query
- The evaluation feedback explaining why retrieval was poor

Rewrite the query to:
1. Be more specific and targeted.
2. Use different vocabulary or synonyms where appropriate.
3. Break down ambiguous terms.
4. Add relevant domain context if missing.

Return ONLY the rewritten query string, nothing else.
"""

# ---------------------------------------------------------------------------
# Query Rewriter node
# ---------------------------------------------------------------------------

def query_rewriter_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph node: rewrite the current query and increment retry counter."""

    current_query = state.get("current_query") or state["original_query"]
    feedback = state.get("evaluation_feedback", "")
    retry_count = state.get("retry_count", 0)

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3)  # slight creativity
    messages = [
        SystemMessage(content=_REWRITER_SYSTEM),
        HumanMessage(
            content=(
                f"Original query: {current_query}\n\n"
                f"Evaluation feedback: {feedback}"
            )
        ),
    ]
    response = llm.invoke(messages)
    rewritten_query = response.content.strip()

    logger.info(
        "Query Rewriter (retry %d): '%s' → '%s'",
        retry_count + 1, current_query, rewritten_query,
    )

    return {
        "current_query": rewritten_query,
        "retry_count": retry_count + 1,
        # Reset agent results so the next supervisor run starts clean.
        "agent_results": [],
        "aggregated_context": "",
        "final_answer": None,
    }
