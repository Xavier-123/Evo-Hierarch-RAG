"""
Middle-Layer DB Agent.

Simulates structured SQL database queries using an in-memory SQLite database
seeded with sample data.  In production, replace ``_run_sql`` with a real
database connection (e.g. SQLAlchemy + your RDS/warehouse).
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import LLM_MODEL, LLM_TEMPERATURE
from src.state import AgentResult, GraphState, SubTask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory SQLite demo database
# ---------------------------------------------------------------------------

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS products (
    id      INTEGER PRIMARY KEY,
    name    TEXT,
    price   REAL,
    stock   INTEGER,
    category TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    id         INTEGER PRIMARY KEY,
    product_id INTEGER,
    quantity   INTEGER,
    total      REAL,
    customer   TEXT,
    order_date TEXT
);
"""

_SEED_DATA = """
INSERT OR IGNORE INTO products VALUES
  (1, 'Laptop Pro',    1299.99, 45,  'Electronics'),
  (2, 'Wireless Mouse',  29.99, 200, 'Accessories'),
  (3, 'USB-C Hub',       49.99, 150, 'Accessories'),
  (4, 'Monitor 4K',    399.99,  30,  'Electronics'),
  (5, 'Keyboard Mech',  89.99, 100,  'Accessories');

INSERT OR IGNORE INTO orders VALUES
  (1, 1, 2, 2599.98, 'Alice',   '2025-01-10'),
  (2, 2, 5,  149.95, 'Bob',     '2025-01-12'),
  (3, 4, 1,  399.99, 'Charlie', '2025-01-15'),
  (4, 3, 3,  149.97, 'Alice',   '2025-01-20'),
  (5, 5, 2,  179.98, 'Dave',    '2025-02-01');
"""


def _get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_CREATE_TABLES)
    conn.executescript(_SEED_DATA)
    return conn


def _run_sql(sql: str) -> str:
    """Execute *sql* against the demo database and return results as a string."""
    try:
        conn = _get_db_connection()
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        if not rows:
            return "Query returned no results."
        header = ", ".join(rows[0].keys())
        body = "\n".join(", ".join(str(v) for v in row) for row in rows)
        return f"{header}\n{body}"
    except Exception as exc:  # noqa: BLE001
        return f"SQL error: {exc}"


# ---------------------------------------------------------------------------
# Agent system prompt
# ---------------------------------------------------------------------------

_DB_SYSTEM = """You are a SQL Database Agent with access to the following tables:
- products (id, name, price, stock, category)
- orders   (id, product_id, quantity, total, customer, order_date)

Given a user question, generate a single valid SQLite SELECT statement that
answers it.  Return ONLY the SQL statement, no commentary, no markdown fences.
"""


# ---------------------------------------------------------------------------
# DB Agent node
# ---------------------------------------------------------------------------

def db_agent_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph node: execute the subtask assigned to the DB agent."""

    subtask: SubTask = _find_subtask(state, "db_agent")
    sub_query = subtask["sub_query"]
    system_prompts: Dict[str, str] = state.get("system_prompts", {})
    system_content = system_prompts.get("db_agent", _DB_SYSTEM)

    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    # Step 1: LLM generates SQL.
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=sub_query),
    ]
    response = llm.invoke(messages)
    sql = response.content.strip().strip("```sql").strip("```").strip()

    # Step 2: Execute SQL against the demo database.
    db_result = _run_sql(sql)

    result: AgentResult = {
        "agent_name": "db_agent",
        "sub_query": sub_query,
        "result": db_result,
        "metadata": {"sql": sql},
    }
    logger.info("DB Agent completed. SQL: %s | Result snippet: %.120s", sql, db_result)

    return {"agent_results": [result]}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _find_subtask(state: GraphState, agent_name: str) -> SubTask:
    for st in state.get("subtasks", []):
        if st["agent"] == agent_name:
            return st
    # Fallback
    return {
        "task_id": "fallback",
        "description": agent_name,
        "agent": agent_name,
        "sub_query": state.get("current_query", state["original_query"]),
    }
