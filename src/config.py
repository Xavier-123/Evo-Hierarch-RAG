"""
Centralized configuration for Evo-Hierarch-RAG.

All tuneable parameters and environment variable bindings live here so that
every other module imports from a single source of truth.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# ---------------------------------------------------------------------------
# Self-improving loop
# ---------------------------------------------------------------------------
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))

# Minimum relevancy score (0–1) needed to skip Query Rewriter
RELEVANCY_THRESHOLD: float = 0.6

# ---------------------------------------------------------------------------
# External API Agent
# ---------------------------------------------------------------------------
API_AGENT_BASE_URL: str = os.getenv("API_AGENT_BASE_URL", "https://api.example.com")

# ---------------------------------------------------------------------------
# Agent routing keys  (must match the values returned by the supervisor)
# ---------------------------------------------------------------------------
AGENT_DB: str = "db_agent"
AGENT_API: str = "api_agent"
AGENT_SKILL: str = "skill_agent"
AGENT_RESEARCH: str = "research_agent"

VALID_AGENTS: frozenset = frozenset({AGENT_DB, AGENT_API, AGENT_SKILL, AGENT_RESEARCH})
