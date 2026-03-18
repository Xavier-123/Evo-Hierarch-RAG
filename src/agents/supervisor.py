"""
Top-Layer Supervisor Agent.

Responsibilities
----------------
1. Parse user intent and assess query complexity.
2. Decompose the query into one or more subtasks (Task Decomposition).
3. Decide which sub-agents to activate (dynamic routing).
4. The result-aggregation and final-answer synthesis steps are handled by the
   separate Aggregator node so that this module stays single-responsibility.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import (
    AGENT_API,
    AGENT_DB,
    AGENT_RESEARCH,
    AGENT_SKILL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    MAX_RETRIES,
    VALID_AGENTS,
)
from src.state import GraphState, SubTask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SUPERVISOR_SYSTEM = """You are the Supervisor Agent of a Hierarchical Agentic RAG system.
Your job is to:
1. Understand the user's query.
2. Determine which specialised agents are needed to answer it.
3. Decompose the query into concrete subtasks, one per agent.

Available agents and their responsibilities:
- db_agent      : Queries a structured SQL/relational database for factual, numeric, or tabular data.
- api_agent     : Calls external/enterprise REST APIs for proprietary or service-specific data.
- skill_agent   : Executes a specific domain skill (e.g., calculation, classification, summarisation).
- research_agent: Searches the internet for real-time, breaking, or open-domain information.

You MUST respond with **only** a valid JSON object (no markdown fences) in this exact schema:
{{
  "routing_decision": ["<agent1>", "<agent2>", ...],
  "subtasks": [
    {{
      "task_id": "t1",
      "description": "<human-readable description>",
      "agent": "<agent_name>",
      "sub_query": "<specific query for this agent>"
    }},
    ...
  ]
}}

Rules:
- Include only the agents that are genuinely needed.
- Each agent in routing_decision must have exactly one subtask entry.
- sub_query should be a self-contained question or instruction for the target agent.
"""

# ---------------------------------------------------------------------------
# Supervisor node
# ---------------------------------------------------------------------------

def supervisor_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph node: analyse the query and produce a routing plan."""

    query = state.get("current_query") or state["original_query"]
    system_prompts: Dict[str, str] = state.get("system_prompts", {})
    few_shot_examples: List[Dict[str, str]] = state.get("few_shot_examples", [])

    # Allow the Prompt Optimizer to override the supervisor's system prompt.
    system_content = system_prompts.get("supervisor", _SUPERVISOR_SYSTEM)

    # Build few-shot block if we have examples.
    few_shot_block = ""
    if few_shot_examples:
        examples = "\n".join(
            f"Query: {ex.get('query','')}\nRouting: {ex.get('routing','')}"
            for ex in few_shot_examples[-5:]  # keep last 5
        )
        few_shot_block = f"\n\nExamples from previous runs:\n{examples}"

    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    messages = [
        SystemMessage(content=system_content + few_shot_block),
        HumanMessage(content=f"User query: {query}"),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    try:
        plan: Dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Supervisor returned non-JSON; attempting recovery. Raw: %s", raw)
        # Fallback: route to research_agent only.
        plan = {
            "routing_decision": [AGENT_RESEARCH],
            "subtasks": [
                {
                    "task_id": "t1",
                    "description": "General research",
                    "agent": AGENT_RESEARCH,
                    "sub_query": query,
                }
            ],
        }

    # Sanitise routing decision against known agents.
    routing: List[str] = [
        a for a in plan.get("routing_decision", []) if a in VALID_AGENTS
    ]
    if not routing:
        routing = [AGENT_RESEARCH]

    subtasks: List[SubTask] = [
        st for st in plan.get("subtasks", []) if st.get("agent") in VALID_AGENTS
    ]
    if not subtasks:
        subtasks = [
            {
                "task_id": "t1",
                "description": "General research",
                "agent": AGENT_RESEARCH,
                "sub_query": query,
            }
        ]

    logger.info("Supervisor routing decision: %s", routing)

    return {
        "routing_decision": routing,
        "subtasks": subtasks,
        "current_query": query,
        "retry_count": state.get("retry_count", 0),
        "max_retries": state.get("max_retries", MAX_RETRIES),
        "agent_results": [],
        "failed_cases": [],
        "few_shot_examples": [],
    }
