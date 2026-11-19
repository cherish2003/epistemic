import unittest

try:
    from langgraph.prebuilt.tool_node import ToolRuntime
    from langgraph._internal._runnable import CONF, CONFIG_KEY_RUNTIME

    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False

if HAS_LANGGRAPH:
    from epistemic import Status, VerifiedBy
    from epistemic.langgraph import epistemic_tool_node


def _get_price(symbol):
    """Return the current price for a symbol."""
    return {"A": 28, "B": 32}[symbol]


def _invoke(node, calls):
    runtime = ToolRuntime(
        state={}, context=None, config={}, stream_writer=None, tool_call_id="", store=None
    )
    return node.invoke(calls, config={CONF: {CONFIG_KEY_RUNTIME: runtime}})


@unittest.skipIf(not HAS_LANGGRAPH, "langgraph not installed")
class TestEpistemicToolNode(unittest.TestCase):
    def test_claim_attached_to_tool_result(self):
        node = epistemic_tool_node([_get_price])
        out = _invoke(
            node,
            [{"name": "_get_price", "args": {"symbol": "A"}, "id": "call-1", "type": "tool_call"}],
        )
        self.assertEqual(out["messages"][0].content, "28")
        claim = node.epistemic_claims["call-1"]
        self.assertEqual(claim.value, "28")
        self.assertEqual(claim.source, "_get_price")
        self.assertEqual(claim.kind, "tool_result")
        self.assertIs(claim.status, Status.ESTIMATED)

    def test_trusted_tools_get_verified_status(self):
        node = epistemic_tool_node([_get_price], trusted_tools=["_get_price"])
        _invoke(
            node,
            [{"name": "_get_price", "args": {"symbol": "A"}, "id": "call-2", "type": "tool_call"}],
        )
        self.assertIs(node.epistemic_claims["call-2"].status, Status.VERIFIED)

    def test_belief_accumulates_claims_per_tool(self):
        node = epistemic_tool_node([_get_price])
        _invoke(
            node,
            [{"name": "_get_price", "args": {"symbol": "A"}, "id": "c1", "type": "tool_call"}],
        )
        _invoke(
            node,
            [{"name": "_get_price", "args": {"symbol": "B"}, "id": "c2", "type": "tool_call"}],
        )
        belief = node.epistemic_beliefs["_get_price"]
        self.assertEqual(len(belief.claims), 2)
        self.assertTrue(belief.has_unresolved_conflicts())

    def test_tool_policy_enforced(self):
        from epistemic import PolicyViolationError

        node = epistemic_tool_node(
            [_get_price],
            tool_policies={"_get_price": (VerifiedBy("trusted_source"),)},
            handle_tool_errors=False,
        )
        with self.assertRaises(PolicyViolationError):
            _invoke(
                node,
                [
                    {
                        "name": "_get_price",
                        "args": {"symbol": "A"},
                        "id": "c3",
                        "type": "tool_call",
                    }
                ],
            )

    def test_gate_check_passes_for_trusted_claim(self):
        from epistemic import EpistemicActionGate

        gate = EpistemicActionGate()
        node = epistemic_tool_node(
            [_get_price], trusted_tools=["_get_price"], action_gate=gate
        )
        gate.register("_get_price", VerifiedBy("_get_price"))
        _invoke(
            node,
            [{"name": "_get_price", "args": {"symbol": "A"}, "id": "c4", "type": "tool_call"}],
        )
        claim = node.epistemic_claims["c4"]
        self.assertEqual(gate.check("_get_price", claim), claim)


if __name__ == "__main__":
    unittest.main()
