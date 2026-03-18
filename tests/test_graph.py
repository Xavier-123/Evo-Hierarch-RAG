"""
Unit tests for Evo-Hierarch-RAG.

These tests use ``unittest.mock`` to avoid requiring live API keys, so they
run entirely offline and in CI without any environment setup.
"""

from __future__ import annotations

import json
import sys
import types
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub out optional heavy dependencies before importing project modules
# ---------------------------------------------------------------------------

def _make_stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


# Stub langchain_openai so tests don't need the package installed.
_loai = _make_stub_module("langchain_openai")
_loai.ChatOpenAI = MagicMock  # type: ignore[attr-defined]

# Stub langchain_core.messages
_lc = _make_stub_module("langchain_core")
_lc_msgs = _make_stub_module("langchain_core.messages")
_lc_msgs.SystemMessage = MagicMock  # type: ignore[attr-defined]
_lc_msgs.HumanMessage = MagicMock   # type: ignore[attr-defined]

# Stub langgraph to avoid needing the real package for pure-unit tests
_lg = _make_stub_module("langgraph")
_lg_graph = _make_stub_module("langgraph.graph")
_lg_graph.END = "END"  # type: ignore[attr-defined]
_lg_graph.START = "START"  # type: ignore[attr-defined]
_lg_types = _make_stub_module("langgraph.types")


# Minimal StateGraph stub that accepts all builder calls without error.
class _MockStateGraph:
    def __init__(self, *args, **kwargs):
        pass

    def add_node(self, *args, **kwargs):
        return self

    def add_edge(self, *args, **kwargs):
        return self

    def add_conditional_edges(self, *args, **kwargs):
        return self

    def compile(self):
        return MagicMock()


_lg_graph.StateGraph = _MockStateGraph  # type: ignore[attr-defined]


# Minimal Send stub: just records (node, state) as a named tuple-like object.
class _MockSend:
    def __init__(self, node: str, state: Any):
        self.node = node
        self.state = state


_lg_types.Send = _MockSend  # type: ignore[attr-defined]

# Stub dotenv
_de = _make_stub_module("dotenv")
_de.load_dotenv = lambda: None  # type: ignore[attr-defined]


class TestGraphState(unittest.TestCase):
    """Verify that GraphState fields are well-defined and reducers work."""

    def test_state_fields_present(self):
        from src.state import GraphState
        annotations = GraphState.__annotations__
        required_fields = [
            "original_query", "current_query", "subtasks", "routing_decision",
            "agent_results", "aggregated_context", "relevancy_score",
            "hallucination_detected", "evaluation_feedback", "retry_count",
            "max_retries", "failed_cases", "system_prompts", "few_shot_examples",
            "final_answer",
        ]
        for field in required_fields:
            self.assertIn(field, annotations, f"Missing field: {field}")

    def test_agent_result_fields(self):
        from src.state import AgentResult
        annotations = AgentResult.__annotations__
        for field in ("agent_name", "sub_query", "result", "metadata"):
            self.assertIn(field, annotations)

    def test_subtask_fields(self):
        from src.state import SubTask
        annotations = SubTask.__annotations__
        for field in ("task_id", "description", "agent", "sub_query"):
            self.assertIn(field, annotations)

    def test_failed_case_fields(self):
        from src.state import FailedCase
        annotations = FailedCase.__annotations__
        for field in ("query", "answer", "feedback", "agent"):
            self.assertIn(field, annotations)


class TestConfig(unittest.TestCase):
    """Verify config defaults and valid-agents set."""

    def test_valid_agents(self):
        from src.config import VALID_AGENTS
        for agent in ("db_agent", "api_agent", "skill_agent", "research_agent"):
            self.assertIn(agent, VALID_AGENTS)

    def test_relevancy_threshold_in_range(self):
        from src.config import RELEVANCY_THRESHOLD
        self.assertGreater(RELEVANCY_THRESHOLD, 0.0)
        self.assertLess(RELEVANCY_THRESHOLD, 1.0)

    def test_max_retries_positive(self):
        from src.config import MAX_RETRIES
        self.assertGreater(MAX_RETRIES, 0)


class TestSupervisorNode(unittest.TestCase):
    """Supervisor node: test routing & fallback logic."""

    def _make_llm_response(self, routing, subtasks):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({"routing_decision": routing, "subtasks": subtasks})
        )
        return mock_llm

    @patch("src.agents.supervisor.ChatOpenAI")
    def test_supervisor_routes_to_valid_agents(self, mock_cls):
        mock_cls.return_value = self._make_llm_response(
            routing=["db_agent"],
            subtasks=[
                {
                    "task_id": "t1",
                    "description": "Query DB",
                    "agent": "db_agent",
                    "sub_query": "SELECT * FROM products",
                }
            ],
        )
        from src.agents.supervisor import supervisor_node

        state: Dict[str, Any] = {
            "original_query": "List all products",
            "current_query": "List all products",
            "system_prompts": {},
            "few_shot_examples": [],
            "retry_count": 0,
            "max_retries": 3,
        }
        result = supervisor_node(state)

        self.assertIn("db_agent", result["routing_decision"])
        self.assertEqual(len(result["subtasks"]), 1)
        self.assertEqual(result["subtasks"][0]["agent"], "db_agent")

    @patch("src.agents.supervisor.ChatOpenAI")
    def test_supervisor_fallback_on_bad_json(self, mock_cls):
        """When LLM returns garbage, supervisor falls back to research_agent."""
        mock_cls.return_value = MagicMock(
            invoke=MagicMock(return_value=MagicMock(content="not valid json at all"))
        )
        from src.agents.supervisor import supervisor_node

        state: Dict[str, Any] = {
            "original_query": "test",
            "current_query": "test",
            "system_prompts": {},
            "few_shot_examples": [],
            "retry_count": 0,
            "max_retries": 3,
        }
        result = supervisor_node(state)
        self.assertIn("research_agent", result["routing_decision"])

    @patch("src.agents.supervisor.ChatOpenAI")
    def test_supervisor_strips_unknown_agents(self, mock_cls):
        mock_cls.return_value = self._make_llm_response(
            routing=["db_agent", "unknown_agent"],
            subtasks=[
                {
                    "task_id": "t1",
                    "description": "Query DB",
                    "agent": "db_agent",
                    "sub_query": "SELECT * FROM products",
                },
                {
                    "task_id": "t2",
                    "description": "Unknown",
                    "agent": "unknown_agent",
                    "sub_query": "something",
                },
            ],
        )
        from src.agents.supervisor import supervisor_node

        state: Dict[str, Any] = {
            "original_query": "test",
            "current_query": "test",
            "system_prompts": {},
            "few_shot_examples": [],
            "retry_count": 0,
            "max_retries": 3,
        }
        result = supervisor_node(state)
        self.assertNotIn("unknown_agent", result["routing_decision"])
        self.assertIn("db_agent", result["routing_decision"])


class TestDBAgent(unittest.TestCase):
    """DB Agent: test SQL generation and in-memory execution."""

    @patch("src.agents.db_agent.ChatOpenAI")
    def test_db_agent_returns_result(self, mock_cls):
        mock_cls.return_value = MagicMock(
            invoke=MagicMock(
                return_value=MagicMock(content="SELECT * FROM products LIMIT 2")
            )
        )
        from src.agents.db_agent import db_agent_node

        state: Dict[str, Any] = {
            "original_query": "Show me products",
            "current_query": "Show me products",
            "subtasks": [
                {
                    "task_id": "t1",
                    "description": "DB query",
                    "agent": "db_agent",
                    "sub_query": "Show me products",
                }
            ],
            "system_prompts": {},
        }
        result = db_agent_node(state)
        agent_results = result["agent_results"]
        self.assertEqual(len(agent_results), 1)
        self.assertEqual(agent_results[0]["agent_name"], "db_agent")
        self.assertIn("Laptop Pro", agent_results[0]["result"])

    @patch("src.agents.db_agent.ChatOpenAI")
    def test_db_agent_handles_bad_sql(self, mock_cls):
        mock_cls.return_value = MagicMock(
            invoke=MagicMock(
                return_value=MagicMock(content="NOT VALID SQL !!!")
            )
        )
        from src.agents.db_agent import db_agent_node

        state: Dict[str, Any] = {
            "original_query": "test",
            "current_query": "test",
            "subtasks": [
                {
                    "task_id": "t1",
                    "description": "DB query",
                    "agent": "db_agent",
                    "sub_query": "test",
                }
            ],
            "system_prompts": {},
        }
        result = db_agent_node(state)
        # Should return an error string, not raise
        self.assertIn("SQL error", result["agent_results"][0]["result"])


class TestAPIAgent(unittest.TestCase):
    """API Agent: verify mock API responses and fallback."""

    @patch("src.agents.api_agent.ChatOpenAI")
    def test_api_agent_mock_users(self, mock_cls):
        mock_cls.return_value = MagicMock(
            invoke=MagicMock(
                return_value=MagicMock(
                    content=json.dumps({"method": "GET", "path": "/users", "params": {}})
                )
            )
        )
        from src.agents.api_agent import api_agent_node

        state: Dict[str, Any] = {
            "original_query": "List all users",
            "current_query": "List all users",
            "subtasks": [
                {
                    "task_id": "t1",
                    "description": "API users",
                    "agent": "api_agent",
                    "sub_query": "List all users",
                }
            ],
            "system_prompts": {},
        }
        result = api_agent_node(state)
        agent_results = result["agent_results"]
        self.assertEqual(agent_results[0]["agent_name"], "api_agent")
        self.assertIn("Alice Chen", agent_results[0]["result"])

    @patch("src.agents.api_agent.ChatOpenAI")
    def test_api_agent_handles_bad_json_response(self, mock_cls):
        mock_cls.return_value = MagicMock(
            invoke=MagicMock(
                return_value=MagicMock(content="not json")
            )
        )
        from src.agents.api_agent import api_agent_node

        state: Dict[str, Any] = {
            "original_query": "test",
            "current_query": "test",
            "subtasks": [
                {
                    "task_id": "t1",
                    "description": "API call",
                    "agent": "api_agent",
                    "sub_query": "test",
                }
            ],
            "system_prompts": {},
        }
        # Should not raise; should fall back to /metrics
        result = api_agent_node(state)
        self.assertIsNotNone(result["agent_results"][0]["result"])


class TestSkillAgent(unittest.TestCase):
    """Skill Agent: verify LLM delegation."""

    @patch("src.agents.skill_agent.ChatOpenAI")
    def test_skill_agent_delegates_to_llm(self, mock_cls):
        mock_cls.return_value = MagicMock(
            invoke=MagicMock(
                return_value=MagicMock(content="42 is the answer")
            )
        )
        from src.agents.skill_agent import skill_agent_node

        state: Dict[str, Any] = {
            "original_query": "What is 6 × 7?",
            "current_query": "What is 6 × 7?",
            "subtasks": [
                {
                    "task_id": "t1",
                    "description": "Math",
                    "agent": "skill_agent",
                    "sub_query": "What is 6 × 7?",
                }
            ],
            "system_prompts": {},
        }
        result = skill_agent_node(state)
        self.assertEqual(result["agent_results"][0]["result"], "42 is the answer")


class TestResearchAgent(unittest.TestCase):
    """Research Agent: verify stub search and LLM synthesis."""

    @patch("src.agents.research_agent.ChatOpenAI")
    @patch("src.agents.research_agent.TAVILY_API_KEY", "")  # force stub mode
    def test_research_agent_uses_stub_without_key(self, mock_cls):
        mock_cls.return_value = MagicMock(
            invoke=MagicMock(
                return_value=MagicMock(content="Here is a synthesised answer.")
            )
        )
        from src.agents.research_agent import research_agent_node

        state: Dict[str, Any] = {
            "original_query": "Latest AI news?",
            "current_query": "Latest AI news?",
            "subtasks": [
                {
                    "task_id": "t1",
                    "description": "Web search",
                    "agent": "research_agent",
                    "sub_query": "Latest AI news?",
                }
            ],
            "system_prompts": {},
        }
        result = research_agent_node(state)
        self.assertEqual(result["agent_results"][0]["agent_name"], "research_agent")
        self.assertEqual(result["agent_results"][0]["result"], "Here is a synthesised answer.")


class TestEvaluatorNode(unittest.TestCase):
    """Evaluator: test scoring, hallucination detection, and routing."""

    def _state(self, score: float, hallucination: bool) -> Dict[str, Any]:
        return {
            "original_query": "test query",
            "aggregated_context": "some context",
            "final_answer": "some answer",
            "retry_count": 0,
            "max_retries": 3,
            "relevancy_score": 0.0,
            "hallucination_detected": False,
            "evaluation_feedback": "",
            "failed_cases": [],
        }

    @patch("src.nodes.evaluator.ChatOpenAI")
    def test_evaluator_high_quality(self, mock_cls):
        mock_cls.return_value = MagicMock(
            invoke=MagicMock(
                return_value=MagicMock(
                    content=json.dumps({
                        "relevancy_score": 0.9,
                        "hallucination_detected": False,
                        "feedback": "Answer is accurate.",
                    })
                )
            )
        )
        from src.nodes.evaluator import evaluator_node, route_after_evaluation

        state = self._state(0.9, False)
        result = evaluator_node(state)
        self.assertAlmostEqual(result["relevancy_score"], 0.9)
        self.assertFalse(result["hallucination_detected"])
        # No failed cases should be recorded
        self.assertEqual(result.get("failed_cases", []), [])

        # Routing should go to "end"
        state.update(result)
        self.assertEqual(route_after_evaluation(state), "end")

    @patch("src.nodes.evaluator.ChatOpenAI")
    def test_evaluator_low_relevancy_routes_to_rewriter(self, mock_cls):
        mock_cls.return_value = MagicMock(
            invoke=MagicMock(
                return_value=MagicMock(
                    content=json.dumps({
                        "relevancy_score": 0.3,
                        "hallucination_detected": False,
                        "feedback": "Context is off-topic.",
                    })
                )
            )
        )
        from src.nodes.evaluator import evaluator_node, route_after_evaluation

        state = self._state(0.3, False)
        result = evaluator_node(state)
        state.update(result)
        self.assertEqual(route_after_evaluation(state), "query_rewriter")

    @patch("src.nodes.evaluator.ChatOpenAI")
    def test_evaluator_hallucination_routes_to_optimizer(self, mock_cls):
        mock_cls.return_value = MagicMock(
            invoke=MagicMock(
                return_value=MagicMock(
                    content=json.dumps({
                        "relevancy_score": 0.8,
                        "hallucination_detected": True,
                        "feedback": "Answer contains made-up facts.",
                    })
                )
            )
        )
        from src.nodes.evaluator import evaluator_node, route_after_evaluation

        state = self._state(0.8, True)
        result = evaluator_node(state)
        state.update(result)
        self.assertEqual(route_after_evaluation(state), "prompt_optimizer")

    def test_evaluator_max_retries_forces_end(self):
        from src.nodes.evaluator import route_after_evaluation

        state: Dict[str, Any] = {
            "retry_count": 5,
            "max_retries": 3,
            "relevancy_score": 0.1,
            "hallucination_detected": True,
        }
        self.assertEqual(route_after_evaluation(state), "end")

    @patch("src.nodes.evaluator.ChatOpenAI")
    def test_evaluator_handles_non_json_response(self, mock_cls):
        mock_cls.return_value = MagicMock(
            invoke=MagicMock(
                return_value=MagicMock(content="I cannot evaluate this right now.")
            )
        )
        from src.nodes.evaluator import evaluator_node

        state = self._state(0.0, False)
        result = evaluator_node(state)
        # Should default to 0.0 / True and not raise
        self.assertEqual(result["relevancy_score"], 0.0)
        self.assertTrue(result["hallucination_detected"])


class TestQueryRewriterNode(unittest.TestCase):
    """Query Rewriter: verify query is rewritten and retry count incremented."""

    @patch("src.nodes.query_rewriter.ChatOpenAI")
    def test_query_rewriter_increments_retry(self, mock_cls):
        mock_cls.return_value = MagicMock(
            invoke=MagicMock(
                return_value=MagicMock(content="improved query text")
            )
        )
        from src.nodes.query_rewriter import query_rewriter_node

        state: Dict[str, Any] = {
            "original_query": "original q",
            "current_query": "original q",
            "evaluation_feedback": "context was irrelevant",
            "retry_count": 1,
        }
        result = query_rewriter_node(state)
        self.assertEqual(result["current_query"], "improved query text")
        self.assertEqual(result["retry_count"], 2)
        self.assertEqual(result["agent_results"], [])
        self.assertIsNone(result["final_answer"])


class TestPromptOptimizerNode(unittest.TestCase):
    """Prompt Optimizer: verify system_prompts are updated."""

    @patch("src.nodes.prompt_optimizer.ChatOpenAI")
    def test_prompt_optimizer_updates_prompts(self, mock_cls):
        mock_cls.return_value = MagicMock(
            invoke=MagicMock(
                return_value=MagicMock(content="NEW improved system prompt")
            )
        )
        from src.nodes.prompt_optimizer import prompt_optimizer_node

        state: Dict[str, Any] = {
            "failed_cases": [
                {
                    "query": "What is the capital?",
                    "answer": "The capital is Paris.",
                    "feedback": "Hallucinated — no context supports this.",
                    "agent": "aggregator",
                }
            ],
            "system_prompts": {"aggregator": "old prompt"},
            "retry_count": 0,
        }
        result = prompt_optimizer_node(state)
        self.assertEqual(result["system_prompts"]["aggregator"], "NEW improved system prompt")
        self.assertEqual(result["retry_count"], 1)
        self.assertIsNone(result["final_answer"])
        self.assertFalse(result["hallucination_detected"])


class TestAggregatorNode(unittest.TestCase):
    """Aggregator: verify it merges multiple agent results."""

    @patch("src.nodes.aggregator.ChatOpenAI")
    def test_aggregator_merges_results(self, mock_cls):
        mock_cls.return_value = MagicMock(
            invoke=MagicMock(
                return_value=MagicMock(content="Merged final answer.")
            )
        )
        from src.nodes.aggregator import aggregator_node

        state: Dict[str, Any] = {
            "original_query": "Combined query",
            "agent_results": [
                {
                    "agent_name": "db_agent",
                    "sub_query": "DB sub-query",
                    "result": "DB result data",
                    "metadata": {},
                },
                {
                    "agent_name": "research_agent",
                    "sub_query": "Research sub-query",
                    "result": "Research result data",
                    "metadata": {},
                },
            ],
            "system_prompts": {},
        }
        result = aggregator_node(state)
        self.assertEqual(result["final_answer"], "Merged final answer.")
        self.assertIn("DB result data", result["aggregated_context"])
        self.assertIn("Research result data", result["aggregated_context"])


class TestGraphFanOut(unittest.TestCase):
    """Graph fan-out router: verify Send objects are created correctly."""

    def test_fan_out_single_agent(self):
        from src.graph import fan_out_to_agents

        state: Dict[str, Any] = {
            "original_query": "test",
            "current_query": "test",
            "routing_decision": ["db_agent"],
            "subtasks": [],
            "agent_results": [],
            "aggregated_context": "",
            "relevancy_score": 0.0,
            "hallucination_detected": False,
            "evaluation_feedback": "",
            "retry_count": 0,
            "max_retries": 3,
            "failed_cases": [],
            "system_prompts": {},
            "few_shot_examples": [],
            "final_answer": None,
        }
        sends = fan_out_to_agents(state)
        self.assertEqual(len(sends), 1)

    def test_fan_out_multiple_agents(self):
        from src.graph import fan_out_to_agents

        state: Dict[str, Any] = {
            "original_query": "test",
            "current_query": "test",
            "routing_decision": ["db_agent", "research_agent"],
            "subtasks": [],
            "agent_results": [],
            "aggregated_context": "",
            "relevancy_score": 0.0,
            "hallucination_detected": False,
            "evaluation_feedback": "",
            "retry_count": 0,
            "max_retries": 3,
            "failed_cases": [],
            "system_prompts": {},
            "few_shot_examples": [],
            "final_answer": None,
        }
        sends = fan_out_to_agents(state)
        self.assertEqual(len(sends), 2)

    def test_fan_out_empty_falls_back_to_research(self):
        from src.graph import fan_out_to_agents

        state: Dict[str, Any] = {
            "original_query": "test",
            "current_query": "test",
            "routing_decision": [],
            "subtasks": [],
            "agent_results": [],
            "aggregated_context": "",
            "relevancy_score": 0.0,
            "hallucination_detected": False,
            "evaluation_feedback": "",
            "retry_count": 0,
            "max_retries": 3,
            "failed_cases": [],
            "system_prompts": {},
            "few_shot_examples": [],
            "final_answer": None,
        }
        sends = fan_out_to_agents(state)
        self.assertEqual(len(sends), 1)


if __name__ == "__main__":
    unittest.main()
