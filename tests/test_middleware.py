import unittest
from datetime import timedelta

from epistemic import (
    Claim,
    EpistemicActionGate,
    Estimated,
    Fresh,
    NoUnresolvedConflicts,
    PolicyViolationError,
    Status,
    Verified,
    VerifiedBy,
    wrap_tool,
)


class TestWrapTool(unittest.TestCase):
    def test_attaches_provenance(self):
        def get_price():
            return 28

        claim = wrap_tool(get_price)()
        self.assertIsInstance(claim, Claim)
        self.assertEqual(claim.value, 28)
        self.assertEqual(claim.source, "get_price")
        self.assertEqual(claim.kind, "tool_result")
        self.assertIs(claim.status, Status.ESTIMATED)

    def test_trusted_status(self):
        def get_price():
            return 28

        claim = wrap_tool(get_price, trusted=True)()
        self.assertIs(claim.status, Status.VERIFIED)
        self.assertEqual(claim.source, "get_price")

    def test_custom_tool_name(self):
        claim = wrap_tool(lambda: 5, tool_name="payment_api")()
        self.assertEqual(claim.source, "payment_api")

    def test_enforces_policies_on_result(self):
        def read_price():
            return 5

        wrapped = wrap_tool(read_price, policies=(VerifiedBy("finance_system"),))
        with self.assertRaises(PolicyViolationError):
            wrapped()

    def test_returns_value_option(self):
        self.assertEqual(wrap_tool(lambda: 5, return_claim=False)(), 5)


class TestEpistemicActionGate(unittest.TestCase):
    def test_no_policies_returns_value(self):
        gate = EpistemicActionGate()
        v = Claim(1, status=Status.VERIFIED)
        self.assertIs(gate.check("payment.execute", v), v)

    def test_enforces_registered_policies(self):
        gate = EpistemicActionGate()
        gate.register("payment.execute", VerifiedBy("finance_system"))
        with self.assertRaises(PolicyViolationError):
            gate.check("payment.execute", Estimated(5))
        gate.check("payment.execute", Verified(5, source="finance_system"))

    def test_multiple_policies_all_enforced(self):
        gate = EpistemicActionGate()
        gate.register("pay", VerifiedBy("finance_system"), Fresh(timedelta(hours=24)))
        with self.assertRaises(PolicyViolationError):
            gate.check("pay", Claim(5, source="finance_system", status=Status.VERIFIED))

    def test_policies_for(self):
        gate = EpistemicActionGate()
        gate.register("a", NoUnresolvedConflicts())
        self.assertEqual(len(gate.policies_for("a")), 1)
        self.assertEqual(gate.policies_for("b"), ())


if __name__ == "__main__":
    unittest.main()
