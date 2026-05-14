"""
Agent behavior probe tests.
Each test exercises a specific behavior requirement from the spec.
"""

import pytest
from agent.state_classifier import AgentState, classify
from agent.guardrails import check_guardrails
from agent.response_parser import parse_response, guardrail_response


class TestStateMachine:
    def test_vague_turn1_is_clarify(self):
        messages = [{"role": "user", "content": "I need some tests."}]
        assert classify(messages) == AgentState.CLARIFY

    def test_vague_turn1_no_role_is_clarify(self):
        messages = [{"role": "user", "content": "What assessments do you have?"}]
        assert classify(messages) == AgentState.CLARIFY

    def test_sufficient_context_is_recommend(self):
        messages = [
            {"role": "user", "content": "I need to hire a mid-level Java developer with 3 years experience."}
        ]
        assert classify(messages) == AgentState.RECOMMEND

    def test_role_and_qualifier_gives_recommend(self):
        messages = [
            {"role": "user", "content": "We're hiring a senior software engineer."},
            {"role": "assistant", "content": "What skills are you looking for?"},
            {"role": "user", "content": "Strong Python and data analysis skills."},
        ]
        assert classify(messages) == AgentState.RECOMMEND

    def test_refine_signal_gives_refine(self):
        messages = [
            {"role": "user", "content": "Java developer, senior level."},
            {"role": "assistant", "content": "Here are recommendations: Java 8 (New)"},
            {"role": "user", "content": "Actually, also add a personality test."},
        ]
        assert classify(messages) == AgentState.REFINE

    def test_compare_signal_gives_compare(self):
        messages = [
            {"role": "user", "content": "Java developer, senior."},
            {"role": "assistant", "content": "Recommendations: Java 8 (New), OPQ32r"},
            {"role": "user", "content": "Can you compare Java 8 vs Verify Interactive Java?"},
        ]
        assert classify(messages) == AgentState.COMPARE

    def test_turn_cap_forces_recommend(self):
        messages = []
        for i in range(7):
            messages.append({"role": "user", "content": "I need tests for something."})
            messages.append({"role": "assistant", "content": "Can you tell me more?"})
        messages.append({"role": "user", "content": "I still need tests."})
        assert classify(messages, max_turns=8) == AgentState.RECOMMEND


class TestResponseParser:
    def _valid_urls(self):
        return {
            "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
            "https://www.shl.com/solutions/products/product-catalog/view/opq32r/",
        }

    def test_valid_json_parsed(self):
        raw = '{"reply": "Here are my recs.", "recommendations": [], "end_of_conversation": false}'
        result = parse_response(raw, self._valid_urls())
        assert result["reply"] == "Here are my recs."
        assert result["recommendations"] == []
        assert result["end_of_conversation"] is False

    def test_json_in_markdown_fence_parsed(self):
        raw = '```json\n{"reply": "Recs.", "recommendations": [], "end_of_conversation": false}\n```'
        result = parse_response(raw, self._valid_urls())
        assert result["reply"] == "Recs."

    def test_non_catalog_url_stripped(self):
        raw = """{
            "reply": "Here.",
            "recommendations": [
                {"name": "Fake Test", "url": "https://fake.com/test", "test_type": ["Cognitive"]}
            ],
            "end_of_conversation": false
        }"""
        result = parse_response(raw, self._valid_urls())
        assert result["recommendations"] == []

    def test_catalog_url_preserved(self):
        raw = """{
            "reply": "Recommended.",
            "recommendations": [
                {
                    "name": "Java 8 (New)",
                    "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
                    "test_type": ["Knowledge & Skills"]
                }
            ],
            "end_of_conversation": false
        }"""
        result = parse_response(raw, self._valid_urls())
        assert len(result["recommendations"]) == 1
        assert result["recommendations"][0]["name"] == "Java 8 (New)"

    def test_max_10_recommendations(self):
        recs = []
        urls = set()
        for i in range(15):
            url = f"https://www.shl.com/solutions/products/product-catalog/view/test-{i}/"
            recs.append({"name": f"Test {i}", "url": url, "test_type": []})
            urls.add(url)
        raw = f'{{"reply": "Many.", "recommendations": {recs}, "end_of_conversation": false}}'
        import json
        raw = json.dumps({"reply": "Many.", "recommendations": recs, "end_of_conversation": False})
        result = parse_response(raw, urls)
        assert len(result["recommendations"]) <= 10

    def test_malformed_json_returns_safe_fallback(self):
        result = parse_response("not json at all {{{", set())
        assert "reply" in result
        assert result["recommendations"] == []
        assert result["end_of_conversation"] is False

    def test_guardrail_response_injection(self):
        resp = guardrail_response("prompt_injection")
        assert resp["recommendations"] == []
        assert resp["end_of_conversation"] is False
        assert "instruction" in resp["reply"].lower() or "not able" in resp["reply"].lower()

    def test_guardrail_response_out_of_scope(self):
        resp = guardrail_response("out_of_scope")
        assert resp["recommendations"] == []
        assert resp["end_of_conversation"] is False
