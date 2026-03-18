"""
Middle-Layer API Agent.

Calls external/enterprise REST APIs to retrieve proprietary or
service-specific data.  In this demo the agent uses ``httpx`` to hit a
configurable base URL.  Because the demo environment has no live backend,
responses are mocked when the URL is the placeholder value.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict
from urllib.parse import urljoin

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import API_AGENT_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_BASE, OPENAI_API_KEY
from src.state import AgentResult, GraphState, SubTask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_API_SYSTEM = """You are an API Agent that knows how to call enterprise REST APIs.

Available endpoints on {base_url}:
  GET /users          — list all users
  GET /users/{{id}}    — get a specific user
  GET /reports/summary — get a business summary report
  GET /inventory      — get current inventory levels
  GET /metrics        — get key performance metrics

Given a user question, respond with **only** a JSON object:
{{
  "method": "GET",
  "path": "/endpoint",
  "params": {{}}
}}
No commentary, no markdown fences.
"""

_API_SYSTEM_ZH = """你是一个知道如何调用企业REST API的API Agent。

Available endpoints on {base_url}:
    GET /users           — 列出所有用户
    GET /users/{{id}}    — 获取特定用户信息
    GET /reports/summary — 获取业务摘要报告
    GET /inventory       — 获取当前库存水平
    GET /metrics         — 获取关键绩效指标

根据用户的问题，仅以JSON对象作为响应：
{{
    "method": "GET",
    "path": "/endpoint",
    "params": {{}}
}}
不要包含任何解释，也不要使用markdown代码块标记。
"""



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOCK_RESPONSES: Dict[str, Any] = {
    "/users": [
        {"id": 1, "name": "Alice Chen",   "role": "Admin"},
        {"id": 2, "name": "Bob Müller",   "role": "Analyst"},
        {"id": 3, "name": "Charlie Park", "role": "Developer"},
    ],
    "/reports/summary": {
        "revenue": 1_250_000,
        "expenses": 980_000,
        "profit": 270_000,
        "quarter": "Q1-2025",
    },
    "/inventory": [
        {"product": "Laptop Pro",    "stock": 45,  "reorder_level": 10},
        {"product": "Wireless Mouse","stock": 200, "reorder_level": 50},
        {"product": "Monitor 4K",    "stock": 30,  "reorder_level": 5},
    ],
    "/metrics": {
        "dau": 12_450,
        "mau": 87_300,
        "conversion_rate": 0.034,
        "churn_rate": 0.012,
    },
}


def _call_api(method: str, path: str, params: Dict[str, Any]) -> str:
    """Return mock or live API response as a formatted string."""
    is_mock = API_AGENT_BASE_URL == "https://api.example.com" or not API_AGENT_BASE_URL

    if is_mock:
        # Strip trailing slash and find best matching mock key.
        normalised = path.rstrip("/")
        for key, value in _MOCK_RESPONSES.items():
            if normalised.startswith(key):
                return json.dumps(value, indent=2)
        return json.dumps({"error": f"No mock data for path '{path}'"})

    # Live HTTP call (requires httpx).
    try:
        import httpx  # lazy import so httpx stays optional in tests

        url = urljoin(API_AGENT_BASE_URL, path)
        with httpx.Client(timeout=10.0) as client:
            resp = client.request(method, url, params=params)
            resp.raise_for_status()
            return resp.text
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# API Agent node
# ---------------------------------------------------------------------------

def api_agent_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph node: execute the subtask assigned to the API agent."""

    subtask: SubTask = _find_subtask(state, "api_agent")
    sub_query = subtask["sub_query"]
    system_prompts: Dict[str, str] = state.get("system_prompts", {})
    system_content = system_prompts.get(
        # "api_agent", _API_SYSTEM.format(base_url=API_AGENT_BASE_URL)
        "api_agent", _API_SYSTEM_ZH.format(base_url=API_AGENT_BASE_URL)
    )

    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE)

    # Step 1: LLM decides which API endpoint to call.
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=sub_query),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip().strip("```json").strip("```").strip()

    try:
        call_spec: Dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        call_spec = {"method": "GET", "path": "/metrics", "params": {}}

    method = call_spec.get("method", "GET")
    path = call_spec.get("path", "/metrics")
    params = call_spec.get("params", {})

    # Step 2: Execute the API call.
    api_result = _call_api(method, path, params)

    result: AgentResult = {
        "agent_name": "api_agent",
        "sub_query": sub_query,
        "result": api_result,
        "metadata": {"method": method, "path": path, "params": params},
    }
    logger.info("API Agent completed. %s %s | Result snippet: %.120s", method, path, api_result)

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
