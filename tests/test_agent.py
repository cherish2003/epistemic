import operator
import unittest
from datetime import timedelta

from epistemic import (
    Belief,
    Claim,
    Conflicting,
    Derivation,
    Estimated,
    Fresh,
    IndependentConfirmation,
    NoUnresolvedConflicts,
    PolicyViolationError,
    Recommendation,
    Status,
    Verified,
    VerifiedBy,
    enforce,
    recommend_action,
    wrap_tool_result,
)


def _utcnow():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


class TestClaim(unittest.TestCase):
    def test_memory_claim_from_spec(self):
        c = Claim(
            value="User prefers cheap flights",
            source="conversation_124",
            observed_at=_utcnow(),
            confidence=0.95,
            expires_at=None,
        )
        self.assertEqual(c.value, "User prefers cheap flights")
        self.assertEqual(c.source, "conversation_124")
        self.assertIs(c.status, Status.ESTIMATED)
        self.assertEqual(c.confidence, 0.95)
        self.assertIsNotNone(c.observed_at)

    def test_claim_kind_and_defaults(self):
        c = Claim(1)
        self.assertEqual(c.kind, "claim")
        self.assertIs(c.status, Status.ESTIMATED)
        self.assertEqual(c.confidence, 0.7)
        self.assertEqual(Claim(1, kind="tool_result").kind, "tool_result")

    def test_claim_explicit_status(self):
        self.assertIs(Claim(1, status=Status.VERIFIED).status, Status.VERIFIED)

    def test_claim_expired_is_stale(self):
        c = Claim(1, status=Status.VERIFIED, expires_at=_utcnow() - timedelta(seconds=1))
        self.assertIs(c.status_at(), Status.STALE)


class TestDerivation(unittest.TestCase):
    def test_derivation_propagates_status(self):
        d = Derivation.from_inputs(5, operator.add, [Verified(2), Verified(3)])
        self.assertEqual(d.value, 5)
        self.assertIs(d.status, Status.VERIFIED)
        self.assertEqual(len(d.inputs), 2)
        self.assertIs(d.op, operator.add)

    def test_derivation_estimated_input(self):
        d = Derivation.from_inputs(2.5, operator.add, [Verified(1.0), Verified(1.5)])
        self.assertIs(d.status, Status.VERIFIED)
        d2 = Derivation.from_inputs(3, operator.add, [Estimated(1.0), Estimated(2.0)])
        self.assertIs(d2.status, Status.ESTIMATED)
        self.assertIsNotNone(d2.confidence)

    def test_derivation_merges_evidence(self):
        from epistemic import Evidence

        d = Derivation.from_inputs(
            5,
            operator.add,
            [
                Verified(2, evidence=[Evidence(source="a")]),
                Verified(3, evidence=[Evidence(source="b")]),
            ],
        )
        self.assertEqual([e.source for e in d.evidence], ["a", "b"])


class TestBelief(unittest.TestCase):
    def test_belief_conflicting_values(self):
        b = Belief(
            [
                Claim(100, source="api_a", status=Status.VERIFIED),
                Claim(85, source="api_b", status=Status.VERIFIED),
                Claim(120, source="api_c", status=Status.VERIFIED),
            ]
        )
        self.assertTrue(b.has_unresolved_conflicts())

    def test_belief_no_conflict_when_consistent(self):
        self.assertFalse(Belief([Claim(100)]).has_unresolved_conflicts())
        self.assertFalse(
            Belief(
                [
                    Claim(100, status=Status.VERIFIED),
                    Claim(100, status=Status.VERIFIED),
                ]
            ).has_unresolved_conflicts()
        )

    def test_belief_best_by_confidence(self):
        b = Belief([Claim(5, confidence=0.9), Claim(7, confidence=0.2)])
        self.assertEqual(b.best().value, 5)
        self.assertIsNone(Belief().best())

    def test_belief_sources_and_values(self):
        b = Belief([Claim(1, source="a"), Claim(2, source="b")])
        self.assertEqual(sorted(b.sources()), ["a", "b"])
        self.assertEqual(b.values(), [1, 2])

    def test_belief_wraps_raw_values(self):
        self.assertEqual(len(Belief([1, 2]).claims), 2)


class TestPolicies(unittest.TestCase):
    def test_verified_by_passes_for_verified_claim_from_source(self):
        c = Claim(50, source="finance_system", status=Status.VERIFIED)
        self.assertTrue(VerifiedBy("finance_system").satisfied(c))

    def test_verified_by_fails_for_estimated_claim(self):
        self.assertFalse(
            VerifiedBy("finance_system").satisfied(Claim(50, source="finance_system"))
        )

    def test_verified_by_fails_for_other_source(self):
        self.assertFalse(VerifiedBy("finance_system").satisfied(Verified(50, source="other")))

    def test_verified_by_on_belief_distinct_sources(self):
        b = Belief(
            [
                Claim(100, source="api_a", status=Status.VERIFIED),
                Claim(100, source="api_b", status=Status.VERIFIED),
            ]
        )
        self.assertTrue(VerifiedBy(["api_a", "api_b"], min_independent=2).satisfied(b))

    def test_independent_confirmation_requires_distinct_sources(self):
        b = Belief(
            [
                Claim(100, source="api_a", status=Status.VERIFIED),
                Claim(100, source="api_a", status=Status.VERIFIED),
                Claim(100, source="api_a", status=Status.VERIFIED),
            ]
        )
        self.assertFalse(IndependentConfirmation(["api_a"], min_sources=2).satisfied(b))

    def test_fresh_requires_expiry_window(self):
        now = _utcnow()
        self.assertTrue(
            Fresh(timedelta(hours=1)).satisfied(
                Claim(1, status=Status.VERIFIED, expires_at=now + timedelta(hours=2))
            )
        )
        self.assertFalse(
            Fresh(timedelta(hours=24)).satisfied(
                Claim(1, status=Status.VERIFIED, expires_at=now + timedelta(hours=2))
            )
        )
        self.assertFalse(
            Fresh(timedelta(hours=1)).satisfied(Claim(1, status=Status.VERIFIED))
        )

    def test_no_unresolved_conflicts(self):
        self.assertTrue(NoUnresolvedConflicts().satisfied(Claim(1)))
        self.assertFalse(
            NoUnresolvedConflicts().satisfied(
                Belief(
                    [
                        Claim(1, status=Status.VERIFIED, source="a"),
                        Claim(2, status=Status.VERIFIED, source="b"),
                    ]
                )
            )
        )
        self.assertFalse(
            NoUnresolvedConflicts().satisfied(Conflicting([Verified(1), Verified(2)]))
        )


class TestEnforce(unittest.TestCase):
    def test_enforce_returns_value_when_passing(self):
        v = Verified(5, source="finance_system")
        self.assertIs(enforce(v, VerifiedBy("finance_system")), v)

    def test_enforce_raises_when_failing(self):
        with self.assertRaises(PolicyViolationError):
            enforce(Estimated(5), VerifiedBy("finance_system"))

    def test_enforce_on_belief_with_conflicts(self):
        b = Belief(
            [
                Claim(1, status=Status.VERIFIED, source="a"),
                Claim(2, status=Status.VERIFIED, source="b"),
            ]
        )
        with self.assertRaises(PolicyViolationError):
            enforce(b, NoUnresolvedConflicts())


class TestWrapToolResult(unittest.TestCase):
    def test_wrap_tool_result_attaches_source(self):
        r = wrap_tool_result("weather_api", 28)
        self.assertIsInstance(r, Claim)
        self.assertEqual(r.value, 28)
        self.assertEqual(r.source, "weather_api")
        self.assertEqual(r.kind, "tool_result")
        self.assertIs(r.status, Status.ESTIMATED)

    def test_preserve_all_tool_claims_in_belief(self):
        b = Belief(
            [
                wrap_tool_result("weather_api", 28),
                wrap_tool_result("web_search", 32),
                wrap_tool_result("cached", 35),
            ]
        )
        self.assertTrue(b.has_unresolved_conflicts())

    def test_safe_execution_requirement_scenario(self):
        invoice_claim = wrap_tool_result("email_attachment_parser", 50000)
        with self.assertRaises(PolicyViolationError):
            enforce(
                invoice_claim,
                VerifiedBy("finance_system"),
                Fresh(within=timedelta(hours=24)),
            )
        trusted = Claim(
            50000,
            source="finance_system",
            status=Status.VERIFIED,
            observed_at=_utcnow(),
            expires_at=_utcnow() + timedelta(hours=48),
        )
        enforce(
            trusted,
            VerifiedBy("finance_system"),
            Fresh(within=timedelta(hours=24)),
        )


class TestRecommendAction(unittest.TestCase):
    def test_none_when_satisfied(self):
        b = Belief([Claim(5, status=Status.VERIFIED, source="api")])
        r = recommend_action(b)
        self.assertEqual(r.kind, "none")

    def test_observe_when_unknown(self):
        b = Belief([Claim(None, status=Status.UNKNOWN, source="search_api")])
        r = recommend_action(b)
        self.assertEqual(r.kind, "observe")
        self.assertIn("search_api", r.options)

    def test_revalidate_when_stale(self):
        b = Belief(
            [
                Claim(
                    5,
                    status=Status.VERIFIED,
                    expires_at=_utcnow() - timedelta(seconds=1),
                )
            ]
        )
        r = recommend_action(b)
        self.assertEqual(r.kind, "revalidate")

    def test_verify_on_unresolved_conflicts(self):
        b = Belief(
            [
                Claim(100, source="api_a", status=Status.VERIFIED),
                Claim(85, source="api_b", status=Status.VERIFIED),
            ]
        )
        r = recommend_action(b)
        self.assertEqual(r.kind, "verify")
        self.assertIn("unresolved conflicts", r.rationale)

    def test_ask_human_when_blocked(self):
        b = Belief([Claim(5, source="finance", status=Status.ESTIMATED)])
        r = recommend_action(b, policies=(VerifiedBy("finance_system"),))
        self.assertEqual(r.kind, "ask_human")

    def test_single_epistemic_value(self):
        v = Claim(5, status=Status.UNKNOWN)
        r = recommend_action(v)
        self.assertEqual(r.kind, "observe")


if __name__ == "__main__":
    unittest.main()
