"""
Bottom-Layer Research Agent.

Uses Tavily Search to retrieve real-time, up-to-date information from the
internet.  Falls back to a stub when the API key is absent (e.g. in tests).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import LLM_MODEL, LLM_TEMPERATURE, TAVILY_API_KEY, OPENAI_API_BASE, OPENAI_API_KEY
from src.state import AgentResult, GraphState, SubTask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_RESEARCH_SYSTEM = """You are a Research Agent with access to real-time web search results.

Given the retrieved search snippets and the user's question, synthesise a
concise, factual answer.  Cite the source URLs where relevant.
"""

_RESEARCH_SYSTEM_ZH = """你是一个能够获取实时网络搜索结果的 Research Agent。

根据检索到的搜索结果片段和用户的问题，整合出一个简洁、基于事实的答案。在相关情况下，引用信息来源的URL。
"""

# ---------------------------------------------------------------------------
# Tavily search helper
# ---------------------------------------------------------------------------

def _tavily_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Run a Tavily search and return a list of result dicts."""
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not set; returning stub search results.")
        return [
            {
                "title": "Stub Result",
                "url": "https://example.com",
                "content": f"(No Tavily key configured) Stub answer for: {query}",
            }
        ]

    try:
        from tavily import TavilyClient  # lazy import

        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query=query, max_results=max_results)
        return response.get("results", [])
    except Exception as exc:  # noqa: BLE001
        logger.error("Tavily search failed: %s", exc)
        return [{"title": "Error", "url": "", "content": str(exc)}]


def _format_search_results(results: List[Dict[str, str]]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        url = r.get("url", "")
        content = r.get("content", "")
        lines.append(f"[{i}] {title}\nURL: {url}\nSnippet: {content}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Research Agent node
# ---------------------------------------------------------------------------

def research_agent_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph node: execute the subtask assigned to the Research agent."""

    subtask: SubTask = _find_subtask(state, "research_agent")
    sub_query = subtask["sub_query"]
    system_prompts: Dict[str, str] = state.get("system_prompts", {})
    # system_content = system_prompts.get("research_agent", _RESEARCH_SYSTEM)
    system_content = system_prompts.get("research_agent", _RESEARCH_SYSTEM_ZH)

    # Step 1: Web search.
    search_results = _tavily_search(sub_query)
    context = _format_search_results(search_results)

    # Step 2: LLM synthesises the answer from search snippets.
    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE)
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(
            content=f"Question: {sub_query}\n\nSearch results:\n{context}"
        ),
    ]
    response = llm.invoke(messages)
    research_result = response.content.strip()

    result: AgentResult = {
        "agent_name": "research_agent",
        "sub_query": sub_query,
        "result": research_result,
        "metadata": {
            "num_results": len(search_results),
            "sources": [r.get("url", "") for r in search_results],
        },
    }
    logger.info("Research Agent completed. Sources: %s", result["metadata"]["sources"])

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
