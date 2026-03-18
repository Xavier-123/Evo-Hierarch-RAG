"""
Middle-Layer Skill Agent.

Handles domain-specific computations and capabilities that do not require
external databases or live APIs — for example: mathematical calculations,
text classification, unit conversion, summarisation, and sentiment analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import LLM_MODEL, LLM_TEMPERATURE
from src.state import AgentResult, GraphState, SubTask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SKILL_SYSTEM = """You are a Skill Agent specialised in domain-specific computations and analysis.

Your capabilities include:
- Mathematical and statistical calculations
- Text classification and tagging
- Unit and currency conversions
- Sentiment analysis
- Short text summarisation
- Structured data extraction from unstructured text
- Step-by-step logical reasoning

Solve the user's request accurately and concisely.  Show your reasoning where helpful.
"""

# ---------------------------------------------------------------------------
# Skill Agent node
# ---------------------------------------------------------------------------

def skill_agent_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph node: execute the subtask assigned to the Skill agent."""

    subtask: SubTask = _find_subtask(state, "skill_agent")
    sub_query = subtask["sub_query"]
    system_prompts: Dict[str, str] = state.get("system_prompts", {})
    system_content = system_prompts.get("skill_agent", _SKILL_SYSTEM)

    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=sub_query),
    ]
    response = llm.invoke(messages)
    skill_result = response.content.strip()

    result: AgentResult = {
        "agent_name": "skill_agent",
        "sub_query": sub_query,
        "result": skill_result,
        "metadata": {"skill": subtask.get("description", "general")},
    }
    logger.info("Skill Agent completed. Result snippet: %.120s", skill_result)

    return {"agent_results": [result]}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _find_subtask(state: GraphState, agent_name: str) -> SubTask:
    for st in state.get("subtasks", []):
        if st["agent"] == agent_name:
            return st
    return {
        "task_id": "fallback",
        "description": agent_name,
        "agent": agent_name,
        "sub_query": state.get("current_query", state["original_query"]),
    }
