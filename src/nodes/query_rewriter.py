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

from src.config import LLM_MODEL, OPENAI_API_BASE, OPENAI_API_KEY
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

_REWRITER_SYSTEM_ZH = """你是检索增强生成系统的 query 改写专家。

初始 query 未能检索到足够相关的上下文内容。

已知：
- 初始 query
- 解释检索失败原因的评价反馈

请重写 query，以达成以下目标：
1. 更加具体和有针对性。
2. 适当使用不同的词汇或同义词。
3. 分解模糊不清的术语。
4. 如果缺少相关领域背景信息，请补充进去。

仅返回重写后的 query 字符串，不要包含任何其他内容。
"""

# ---------------------------------------------------------------------------
# Query Rewriter node
# ---------------------------------------------------------------------------

def query_rewriter_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph node: rewrite the current query and increment retry counter."""

    current_query = state.get("current_query") or state["original_query"]
    feedback = state.get("evaluation_feedback", "")
    retry_count = state.get("retry_count", 0)

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3, openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE)  # slight creativity
    messages = [
        # SystemMessage(content=_REWRITER_SYSTEM),
        SystemMessage(content=_REWRITER_SYSTEM_ZH),
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
