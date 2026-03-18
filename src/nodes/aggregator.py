"""
Aggregator Node.

Merges the outputs produced by all active sub-agents into a single coherent
context string and then synthesises the final answer using an LLM.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import LLM_MODEL, LLM_TEMPERATURE
from src.state import AgentResult, GraphState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_AGGREGATOR_SYSTEM = """You are the Result Aggregator of a Hierarchical Agentic RAG system.

You receive a user query and the outputs from one or more specialised agents
(database queries, API calls, skill computations, web research).

Your task:
1. Synthesise all agent outputs into a single, coherent, well-structured answer.
2. Resolve any contradictions between agent outputs by favouring the most specific source.
3. Be concise but complete — include all key facts returned by the agents.
4. If relevant, attribute which source provided which piece of information.
"""


def _format_agent_results(results: list[AgentResult]) -> str:
    parts = []
    for r in results:
        parts.append(
            f"--- {r['agent_name'].upper()} ---\n"
            f"Sub-query: {r['sub_query']}\n"
            f"Result:\n{r['result']}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Aggregator node
# ---------------------------------------------------------------------------

def aggregator_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph node: merge agent results and generate the final answer."""

    agent_results = state.get("agent_results", [])
    original_query = state["original_query"]
    system_prompts: Dict[str, str] = state.get("system_prompts", {})
    system_content = system_prompts.get("aggregator", _AGGREGATOR_SYSTEM)

    aggregated_context = _format_agent_results(agent_results)

    if not agent_results:
        aggregated_context = "No agent results were produced."

    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(
            content=(
                f"User query: {original_query}\n\n"
                f"Agent outputs:\n{aggregated_context}"
            )
        ),
    ]
    response = llm.invoke(messages)
    final_answer = response.content.strip()

    logger.info("Aggregator produced final answer (%.80s…)", final_answer)

    return {
        "aggregated_context": aggregated_context,
        "final_answer": final_answer,
    }
