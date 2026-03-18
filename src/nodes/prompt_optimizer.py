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

from src.config import LLM_MODEL, OPENAI_API_BASE, OPENAI_API_KEY
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

_OPTIMIZER_SYSTEM_ZH = """你是检索增强生成管道的系统提示词优化智能体。

你将接收：
1. 某智能体当前使用的系统提示词。
2. 一系列失败案例：导致该智能体生成幻觉内容的查询。

你的任务：
- 分析常见的失败模式。
- 生成一个改进后的系统提示词，明确阻止这些失败模式的发生。
- 添加1-2个少量示例（query → ideal behaviour）来强化正确行为。
- 保留智能体的核心能力。

仅返回新的系统提示词文本。不要包含 meta-commentary。
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

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2, openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE)
    messages = [
        # SystemMessage(content=_OPTIMIZER_SYSTEM),
        SystemMessage(content=_OPTIMIZER_SYSTEM_ZH),
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
