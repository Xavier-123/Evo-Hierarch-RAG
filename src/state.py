"""
LangGraph state definition for Evo-Hierarch-RAG.

Every node reads from and writes into a single ``GraphState`` TypedDict.
Fields that are accumulated across parallel branches use
``Annotated[List[…], operator.add]`` so that LangGraph merges sub-graph
outputs correctly via the built-in list-concatenation reducer.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


# ---------------------------------------------------------------------------
# Sub-types
# ---------------------------------------------------------------------------

class SubTask(TypedDict):
    """A single decomposed subtask produced by the Supervisor."""

    task_id: str
    description: str
    agent: str          # one of: db_agent | api_agent | skill_agent | research_agent
    sub_query: str      # specialised query sent to the target agent


class AgentResult(TypedDict):
    """Result produced by any Middle- or Bottom-layer agent."""

    agent_name: str
    sub_query: str
    result: str
    metadata: Dict[str, Any]


class FailedCase(TypedDict):
    """Record of a single failed generation, used by the Prompt Optimizer."""

    query: str
    answer: str
    feedback: str
    agent: str


# ---------------------------------------------------------------------------
# Main graph state
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    """Shared state propagated through every node of the LangGraph workflow."""

    # ── Input ──────────────────────────────────────────────────────────────
    original_query: str
    current_query: str          # may be rewritten by QueryRewriter

    # ── Task decomposition (set by Supervisor) ──────────────────────────────
    subtasks: List[SubTask]
    routing_decision: List[str]  # agent names selected for this query

    # ── Agent outputs (accumulated across parallel branches) ────────────────
    agent_results: Annotated[List[AgentResult], operator.add]

    # ── Aggregated context ──────────────────────────────────────────────────
    aggregated_context: str

    # ── Self-improving loop ─────────────────────────────────────────────────
    relevancy_score: float
    hallucination_detected: bool
    evaluation_feedback: str

    retry_count: int
    max_retries: int

    failed_cases: Annotated[List[FailedCase], operator.add]

    # ── Dynamic prompts (keyed by agent name) ───────────────────────────────
    system_prompts: Dict[str, str]
    few_shot_examples: Annotated[List[Dict[str, str]], operator.add]

    # ── Final answer ────────────────────────────────────────────────────────
    final_answer: Optional[str]
