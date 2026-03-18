"""
Prompt Optimizer Node — Self-Improving Component #3.

Triggered when the Evaluator detects hallucinations in the generated answer.
Uses Few-shot Prompt Optimization: it analyzes the failed cases recorded in
the state and generates an improved system prompt for the relevant agent so
that future runs are less likely to hallucinate.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import LLM_MODEL, LLM_TEMPERATURE
from src.state import FailedCase, GraphState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_OPTIMIZER_SYSTEM = """You are a System Prompt Optimizer for a RAG pipeline.

You receive:
1. The current system prompt used by an agent.
2. A list of failed cases: queries where the agent produced hallucinated answers.

Your task:
- Analyse the common failure patterns.
- Produce an improved system prompt that explicitly discourages those failure modes.
- Add 1–2 few-shot examples (query → ideal behaviour) to reinforce correct behaviour.
- Preserve the agent's core capabilities.

Return ONLY the new system prompt text.  Do NOT include meta-commentary.
"""

# ---------------------------------------------------------------------------
# Prompt Optimizer node
# ---------------------------------------------------------------------------

def prompt_optimizer_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph node: update system prompts based on hallucination failures."""

    failed_cases = state.get("failed_cases", [])
    system_prompts: Dict[str, str] = dict(state.get("system_prompts", {}))
    retry_count = state.get("retry_count", 0)

    # Determine which agent to improve.  Default to "aggregator" if unclear.
    target_agent = "aggregator"
    if failed_cases:
        target_agent = failed_cases[-1].get("agent", "aggregator")

    current_prompt = system_prompts.get(target_agent, "")

    # Build failure summary.
    failure_summary = "\n".join(
        f"- Query: {fc.get('query', '')}\n  Feedback: {fc.get('feedback', '')}"
        for fc in failed_cases[-5:]  # use last 5 failures
    )

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)
    messages = [
        SystemMessage(content=_OPTIMIZER_SYSTEM),
        HumanMessage(
            content=(
                f"Current system prompt for '{target_agent}':\n{current_prompt}\n\n"
                f"Recent failures:\n{failure_summary}"
            )
        ),
    ]
    response = llm.invoke(messages)
    improved_prompt = response.content.strip()

    system_prompts[target_agent] = improved_prompt
    logger.info("Prompt Optimizer updated system prompt for '%s'.", target_agent)

    # Build a few-shot example from the last failure so the supervisor benefits.
    few_shot_example: Dict[str, str] = {}
    if failed_cases:
        last = failed_cases[-1]
        few_shot_example = {
            "query": last.get("query", ""),
            "routing": f"Improved routing after hallucination: {last.get('feedback', '')}",
        }

    return {
        "system_prompts": system_prompts,
        "retry_count": retry_count + 1,
        # Reset pipeline for retry.
        "agent_results": [],
        "aggregated_context": "",
        "final_answer": None,
        "hallucination_detected": False,
        "few_shot_examples": [few_shot_example] if few_shot_example else [],
    }
