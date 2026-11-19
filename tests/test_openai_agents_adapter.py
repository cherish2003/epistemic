import asyncio
import unittest
from types import SimpleNamespace

try:
    import agents

    HAS_AGENTS = True
except ImportError:
    HAS_AGENTS = False

if HAS_AGENTS:
    from agents import FunctionTool

    from epistemic import (
        Belief,
        EpistemicActionGate,
        PolicyViolationError,
        Status,
        VerifiedBy,
    )
    from epistemic.openai_agents import EpistemicTools


def _make_tool(name, result):
    async def on_invoke_tool(ctx, args_json):
        return result

    return FunctionTool(
        name=name,
        description="test tool",
        params_json_schema={"type": "object", "properties": {}, "required": []},
        on_invoke_tool=on_invoke_tool,
    )


@unittest.skipIf(not HAS_AGENTS, "openai-agents not installed")
class TestEpistemicToolsAdapter(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_claim_attached_to_tool_result(self):
        adapter = EpistemicTools()
        tool = adapter.wrap_tool(_make_tool("search_api", "result"))
        ctx = SimpleNamespace(tool_call_id="call-1")
        out = self._run(tool.on_invoke_tool(ctx, "{}"))
        self.assertEqual(out, "result")
        claim = adapter.claims_by_call_id["call-1"]
        self.assertEqual(claim.value, "result")
        self.assertEqual(claim.source, "search_api")
        self.assertEqual(claim.kind, "tool_result")
        self.assertIs(claim.status, Status.ESTIMATED)

    def test_trusted_tool_gets_verified_status(self):
        adapter = EpistemicTools(trusted_tools=["search_api"])
        tool = adapter.wrap_tool(_make_tool("search_api", "result"))
        self._run(tool.on_invoke_tool(SimpleNamespace(tool_call_id="call-2"), "{}"))
        self.assertIs(adapter.claims_by_call_id["call-2"].status, Status.VERIFIED)

    def test_beliefs_accumulate_claims_per_tool(self):
        adapter = EpistemicTools()
        tool = adapter.wrap_tool(_make_tool("search_api", "a"))
        self._run(tool.on_invoke_tool(SimpleNamespace(tool_call_id="c1"), "{}"))
        self._run(tool.on_invoke_tool(SimpleNamespace(tool_call_id="c2"), "{}"))
        self.assertEqual(len(adapter.beliefs["search_api"].claims), 2)

    def test_tool_policy_enforced(self):
        adapter = EpistemicTools(
            tool_policies={"search_api": (VerifiedBy("other_source"),)}
        )
        tool = adapter.wrap_tool(_make_tool("search_api", "result"))
        with self.assertRaises(PolicyViolationError):
            self._run(tool.on_invoke_tool(SimpleNamespace(tool_call_id="c3"), "{}"))

    def test_gate_check_with_trusted_claim(self):
        gate = EpistemicActionGate()
        adapter = EpistemicTools(
            trusted_tools=["search_api"], action_gate=gate
        )
        gate.register("search_api", VerifiedBy("search_api"))
        tool = adapter.wrap_tool(_make_tool("search_api", "result"))
        self._run(tool.on_invoke_tool(SimpleNamespace(tool_call_id="c4"), "{}"))
        claim = adapter.claims_by_call_id["c4"]
        self.assertEqual(gate.check("search_api", claim), claim)


if __name__ == "__main__":
    unittest.main()
