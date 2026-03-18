"""
Evaluator Node — Self-Improving Component #1.

Performs two checks on every completed pipeline run:

1. **Relevancy Score**: Does the aggregated context actually address the
   user's original query?  Scored 0.0–1.0.

2. **Hallucination Check**: Does the final answer introduce claims that are
   not supported by the retrieved context?

The node writes back ``relevancy_score``, ``hallucination_detected``, and
``evaluation_feedback`` so that the graph's conditional edge can decide
whether to accept the answer or trigger a retry.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import LLM_MODEL, LLM_TEMPERATURE, RELEVANCY_THRESHOLD, OPENAI_API_BASE, OPENAI_API_KEY
from src.state import FailedCase, GraphState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_EVALUATOR_SYSTEM = """You are a strict Quality Evaluator for a RAG (Retrieval-Augmented Generation) system.

You will receive:
- The user's original query
- The retrieved context (from specialised agents)
- The final generated answer

You must respond with **only** a valid JSON object (no markdown fences):
{{
  "relevancy_score": <float 0.0–1.0>,
  "hallucination_detected": <true|false>,
  "feedback": "<brief explanation>"
}}

Scoring guide:
- relevancy_score >= 0.8 : The context directly and comprehensively answers the query.
- relevancy_score 0.6–0.79: The context partially addresses the query.
- relevancy_score < 0.6  : The context is largely irrelevant.

hallucination_detected = true if the answer contains claims NOT supported by the context.
"""


_EVALUATOR_SYSTEM_ZH = """你是检索增强生成系统的严格质量评估智能体。

你将接收以下信息：
- 用户的原始查询
- 检索到的上下文内容（来自各专业智能体）
- 最终生成的答案

你必须仅以有效的JSON对象（无需markdown代码块标记）作为响应，格式如下：
{{
"relevancy_score": <浮点数 0.0–1.0>,
"hallucination_detected": <true|false>,
"feedback": "<简要说明>"
}}

评分标准：
- relevancy_score >= 0.8 ：上下文内容直接且全面地回答了查询。
- relevancy_score 0.6–0.79：上下文内容部分地回答了查询。
- relevancy_score < 0.6 ：上下文内容基本与查询无关。

hallucination_detected 为 true 的条件：答案中包含上下文内容所不支持的表述。
"""

# ---------------------------------------------------------------------------
# Evaluator node
# ---------------------------------------------------------------------------

def evaluator_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph node: score relevancy and detect hallucinations."""

    original_query = state["original_query"]
    context = state.get("aggregated_context", "")
    final_answer = state.get("final_answer", "")

    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE)
    messages = [
        # SystemMessage(content=_EVALUATOR_SYSTEM),
        SystemMessage(content=_EVALUATOR_SYSTEM_ZH),
        HumanMessage(
            content=(
                f"Original query: {original_query}\n\n"
                f"Retrieved context:\n{context}\n\n"
                f"Generated answer:\n{final_answer}"
            )
        ),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()

    try:
        evaluation: Dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Evaluator returned non-JSON; defaulting to low quality. Raw: %s", raw)
        evaluation = {
            "relevancy_score": 0.0,
            "hallucination_detected": True,
            "feedback": "Could not parse evaluator response.",
        }

    relevancy_score: float = float(evaluation.get("relevancy_score", 0.0))
    hallucination: bool = bool(evaluation.get("hallucination_detected", False))
    feedback: str = str(evaluation.get("feedback", ""))

    logger.info(
        "Evaluator: relevancy=%.2f  hallucination=%s  feedback=%s",
        relevancy_score, hallucination, feedback,
    )

    updates: Dict[str, Any] = {
        "relevancy_score": relevancy_score,
        "hallucination_detected": hallucination,
        "evaluation_feedback": feedback,
    }

    # Record failed cases for the Prompt Optimizer if quality is low.
    is_failing = relevancy_score < RELEVANCY_THRESHOLD or hallucination
    if is_failing:
        failed_case: FailedCase = {
            "query": original_query,
            "answer": final_answer,
            "feedback": feedback,
            "agent": "aggregator",
        }
        updates["failed_cases"] = [failed_case]

    return updates


# ---------------------------------------------------------------------------
# Routing helper (used by graph.py conditional edge)
# ---------------------------------------------------------------------------

def route_after_evaluation(state: GraphState) -> str:
    """Return the name of the next node after evaluation.

    Returns
    -------
    "query_rewriter"     — relevancy too low, rewrite the query and retry.
    "prompt_optimizer"   — hallucination detected, update prompts then retry.
    "end"                — quality is acceptable, output the final answer.
    """
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if retry_count >= max_retries:
        logger.info("Max retries (%d) reached; accepting answer as-is.", max_retries)
        return "end"

    relevancy_score = state.get("relevancy_score", 1.0)
    hallucination = state.get("hallucination_detected", False)

    if hallucination:
        return "prompt_optimizer"
    if relevancy_score < RELEVANCY_THRESHOLD:
        return "query_rewriter"
    return "end"
