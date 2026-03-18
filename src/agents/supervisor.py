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
    OPENAI_API_KEY,
    OPENAI_API_BASE,
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

_SUPERVISOR_SYSTEM_ZH = """你是分层智能检索增强生成系统的监管智能体。
你的职责是：
1. 理解用户的查询内容。
2. 确定需要哪些专业智能体来回答问题。
3. 将查询分解为具体的子任务，每个子任务对应一个智能体。

可用智能体及其职责：
- db_agent：查询结构化SQL/关系型数据库，获取事实性、数值性或表格数据。
- api_agent：调用外部/企业内部REST API，获取专有或特定服务的数据。
- skill_agent：执行特定领域技能（如计算、分类、总结等）。
- research_agent：搜索互联网，获取实时、突发或开放领域信息。

你必须仅以有效的JSON对象（无需markdown代码块标记）作为响应，格式必须严格符合以下结构：
{{
"routing_decision": ["<智能体名称1>", "<智能体名称2>", ...],
"subtasks": [
{{
"task_id": "t1",
"description": "<人类可读的任务描述>",
"agent": "<智能体名称>",
"sub_query": "<针对该智能体的具体查询>"
}},
...
]
}}

规则：
- 只包含真正需要的智能体。
- routing_decision中的每个智能体必须对应一个且只有一个子任务。
- sub_query应当是一个针对目标智能体的、可以独立处理的问题或指令。
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
    # system_content = system_prompts.get("supervisor", _SUPERVISOR_SYSTEM)
    system_content = system_prompts.get("supervisor", _SUPERVISOR_SYSTEM_ZH)

    # Build few-shot block if we have examples.
    few_shot_block = ""
    if few_shot_examples:
        examples = "\n".join(
            f"Query: {ex.get('query','')}\nRouting: {ex.get('routing','')}"
            for ex in few_shot_examples[-5:]  # keep last 5
        )
        few_shot_block = f"\n\nExamples from previous runs:\n{examples}"

    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE)  # Use default OpenAI base URL

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
