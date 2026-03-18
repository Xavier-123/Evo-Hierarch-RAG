"""
LangGraph StateGraph for Evo-Hierarch-RAG.

Architecture
============
┌─────────────────────────────────────────────────────────────┐
│  Top Layer          Supervisor (router + task decomposer)   │
├─────────────────────────────────────────────────────────────┤
│  Middle Layer       DB Agent | API Agent | Skill Agent      │
├─────────────────────────────────────────────────────────────┤
│  Bottom Layer       Research Agent (Tavily web search)      │
├─────────────────────────────────────────────────────────────┤
│  Self-Improving     Aggregator → Evaluator                  │
│  Loop               ├─ [low relevancy] → Query Rewriter     │
│                     └─ [hallucination] → Prompt Optimizer   │
└─────────────────────────────────────────────────────────────┘

Parallel dispatch
-----------------
After the Supervisor decides which agents to activate, the graph fans out to
those agents **in parallel** using LangGraph's ``Send`` API.  All results
accumulate in ``state["agent_results"]`` via the list-concatenation reducer
defined in ``state.py``.
"""

from __future__ import annotations

import logging
from typing import Any, List

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.agents.api_agent import api_agent_node
from src.agents.db_agent import db_agent_node
from src.agents.research_agent import research_agent_node
from src.agents.skill_agent import skill_agent_node
from src.agents.supervisor import supervisor_node
from src.nodes.aggregator import aggregator_node
from src.nodes.evaluator import evaluator_node, route_after_evaluation
from src.nodes.prompt_optimizer import prompt_optimizer_node
from src.nodes.query_rewriter import query_rewriter_node
from src.state import GraphState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent fan-out router
# ---------------------------------------------------------------------------

_AGENT_NODE_MAP = {
    "db_agent":       "db_agent",
    "api_agent":      "api_agent",
    "skill_agent":    "skill_agent",
    "research_agent": "research_agent",
}


def fan_out_to_agents(state: GraphState) -> List[Send]:
    """Return a list of ``Send`` objects — one per active agent.

    LangGraph executes all ``Send`` calls in parallel and merges results back
    into the shared state using the reducer defined on ``agent_results``.
    """
    routing_decision: List[str] = state.get("routing_decision", [])
    sends: List[Send] = []

    for agent_name in routing_decision:
        node_name = _AGENT_NODE_MAP.get(agent_name)
        if node_name:
            sends.append(Send(node_name, state))
        else:
            logger.warning("Unknown agent in routing decision: %s", agent_name)

    if not sends:
        # Fallback: always run research_agent.
        sends.append(Send("research_agent", state))

    return sends


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build and return the compiled LangGraph ``StateGraph``."""

    builder = StateGraph(GraphState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    builder.add_node("supervisor",       supervisor_node)
    builder.add_node("db_agent",         db_agent_node)
    builder.add_node("api_agent",        api_agent_node)
    builder.add_node("skill_agent",      skill_agent_node)
    builder.add_node("research_agent",   research_agent_node)
    builder.add_node("aggregator",       aggregator_node)
    builder.add_node("evaluator",        evaluator_node)
    builder.add_node("query_rewriter",   query_rewriter_node)
    builder.add_node("prompt_optimizer", prompt_optimizer_node)

    # ── Entry edge ──────────────────────────────────────────────────────────
    builder.add_edge(START, "supervisor")

    # ── Supervisor → parallel agent fan-out ────────────────────────────────
    builder.add_conditional_edges(
        "supervisor",
        fan_out_to_agents,
        # The set of possible destination nodes (for graph validation).
        ["db_agent", "api_agent", "skill_agent", "research_agent"],
    )

    # ── All agent nodes → aggregator ────────────────────────────────────────
    for agent_node in ("db_agent", "api_agent", "skill_agent", "research_agent"):
        builder.add_edge(agent_node, "aggregator")

    # ── Aggregator → evaluator ──────────────────────────────────────────────
    builder.add_edge("aggregator", "evaluator")

    # ── Evaluator → conditional routing (self-improving loop) ───────────────
    builder.add_conditional_edges(
        "evaluator",
        route_after_evaluation,
        {
            "query_rewriter":   "query_rewriter",
            "prompt_optimizer": "prompt_optimizer",
            "end":              END,
        },
    )

    # ── Self-improving nodes loop back to supervisor for retry ───────────────
    builder.add_edge("query_rewriter",   "supervisor")
    builder.add_edge("prompt_optimizer", "supervisor")

    return builder.compile()


# ---------------------------------------------------------------------------
# Convenience: pre-built graph instance
# ---------------------------------------------------------------------------

graph = build_graph()
