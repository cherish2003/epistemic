import unittest
import operator
from datetime import datetime, timedelta, timezone

from epistemic import (
    Conflicting,
    Evidence,
    Epistemic,
    Estimated,
    Status,
    Stale,
    UncertainValueError,
    Unknown,
    Verified,
    check_sensitive,
    combine,
)


def _utcnow():
    return datetime.now(timezone.utc)


class TestStatus(unittest.TestCase):
    def test_precedence_order(self):
        self.assertLess(Status.VERIFIED.precedence, Status.STALE.precedence)
        self.assertLess(Status.STALE.precedence, Status.ESTIMATED.precedence)
        self.assertLess(Status.ESTIMATED.precedence, Status.UNKNOWN.precedence)
        self.assertLess(Status.UNKNOWN.precedence, Status.CONFLICTING.precedence)


class TestBaseClass(unittest.TestCase):
    def test_base_epistemic(self):
        v = Epistemic(3, Status.VERIFIED)
        self.assertEqual(v.value, 3)
        self.assertIs(v.status, Status.VERIFIED)
        self.assertEqual(v.confidence, 1.0)
        self.assertIsNone(v.source)
        self.assertEqual(v.evidence, [])
        self.assertEqual(v.alternatives, [])


class TestConstructors(unittest.TestCase):
    def test_verified(self):
        v = Verified(500)
        self.assertEqual(v.value, 500)
        self.assertIs(v.status, Status.VERIFIED)
        self.assertEqual(v.confidence, 1.0)

    def test_stale(self):
        v = Stale(5)
        self.assertIs(v.status, Status.STALE)
        self.assertEqual(v.confidence, 0.8)
        self.assertEqual(v.value, 5)

    def test_estimated_default_confidence(self):
        self.assertEqual(Estimated(0.72).confidence, 0.5)
        self.assertEqual(Estimated(0.72, confidence=0.9).confidence, 0.9)

    def test_unknown(self):
        v = Unknown()
        self.assertIs(v.status, Status.UNKNOWN)
        self.assertIsNone(v.value)
        self.assertEqual(v.confidence, 0.0)

    def test_conflicting(self):
        v = Conflicting([Verified(1), Estimated(2)])
        self.assertIs(v.status, Status.CONFLICTING)
        self.assertEqual(len(v.alternatives), 2)


class TestExpiry(unittest.TestCase):
    def test_expired_is_stale(self):
        v = Verified(5, expires_at=_utcnow() - timedelta(seconds=1))
        self.assertTrue(v.is_expired())
        self.assertFalse(v.is_current())
        self.assertIs(v.status_at(), Status.STALE)

    def test_not_expired(self):
        v = Verified(5, expires_at=_utcnow() + timedelta(hours=1))
        self.assertFalse(v.is_expired())
        self.assertTrue(v.is_current())
        self.assertIs(v.status_at(), Status.VERIFIED)

    def test_no_expiry(self):
        self.assertFalse(Verified(5).is_expired())

    def test_expiration_alias(self):
        v = Verified(5, expiration=_utcnow() + timedelta(hours=1))
        self.assertIsNotNone(v.expires_at)


class TestEvidence(unittest.TestCase):
    def test_evidence_attachment(self):
        e = Evidence(source="api", type="observation", timestamp=_utcnow(), description="obs")
        v = Verified(1, evidence=[e])
        self.assertEqual(v.evidence, [e])
        self.assertEqual(v.evidence[0].source, "api")
        self.assertEqual(v.evidence[0].type, "observation")

    def test_evidence_default_empty(self):
        self.assertEqual(Verified(1).evidence, [])

    def test_source_attachment(self):
        v = Verified(1, source="payment-api")
        self.assertEqual(v.source, "payment-api")


class TestCombine(unittest.TestCase):
    def test_verified_plus_verified(self):
        v = combine(Verified(2), Verified(3))
        self.assertEqual(v.value, 5)
        self.assertIs(v.status, Status.VERIFIED)
        self.assertEqual(v.confidence, 1.0)

    def test_verified_plus_estimated_propagates(self):
        v = combine(Verified(2), Estimated(3))
        self.assertEqual(v.value, 5)
        self.assertIs(v.status, Status.ESTIMATED)

    def test_stale_propagates(self):
        v = combine(Stale(1), Verified(2))
        self.assertIs(v.status, Status.STALE)
        self.assertEqual(v.value, 3)

    def test_expired_propagates_as_stale(self):
        v = combine(Verified(1, expires_at=_utcnow() - timedelta(seconds=1)), Verified(2))
        self.assertIs(v.status, Status.STALE)

    def test_unknown_propagates(self):
        v = combine(Verified(1), Unknown())
        self.assertIs(v.status, Status.UNKNOWN)

    def test_conflicting_propagates(self):
        alternatives = [Verified(1), Verified(2)]
        v = combine(Verified(0), Conflicting(alternatives))
        self.assertIs(v.status, Status.CONFLICTING)
        self.assertEqual(len(v.alternatives), 3)

    def test_raw_values_wrapped_as_verified(self):
        v = combine(5, Verified(2))
        self.assertEqual(v.value, 7)
        self.assertIs(v.status, Status.VERIFIED)

    def test_confidence_min(self):
        v = combine(Estimated(1, confidence=0.6), Estimated(2, confidence=0.8))
        self.assertIs(v.status, Status.ESTIMATED)
        self.assertAlmostEqual(v.confidence, 0.6)

    def test_combine_with_op_sub(self):
        v = combine(Verified(10), Verified(4), op=operator.sub)
        self.assertEqual(v.value, 6)

    def test_requires_values(self):
        with self.assertRaises(ValueError):
            combine()

    def test_evidence_merges(self):
        a = Verified(1, evidence=[Evidence(source="a")])
        b = Verified(2, evidence=[Evidence(source="b")])
        v = combine(a, b)
        self.assertEqual([e.source for e in v.evidence], ["a", "b"])


class TestOperators(unittest.TestCase):
    def test_add_operator(self):
        v = Verified(2) + Verified(3)
        self.assertEqual(v.value, 5)
        self.assertIs(v.status, Status.VERIFIED)

    def test_add_operator_with_estimated(self):
        v = Verified(2) + Estimated(3)
        self.assertIs(v.status, Status.ESTIMATED)

    def test_radd(self):
        v = 5 + Verified(2)
        self.assertEqual(v.value, 7)
        self.assertIs(v.status, Status.VERIFIED)


class TestCheckSensitive(unittest.TestCase):
    def test_verified_passes(self):
        self.assertEqual(check_sensitive(Verified(5)), 5)

    def test_estimated_raises(self):
        with self.assertRaises(UncertainValueError):
            check_sensitive(Estimated(5))

    def test_confidence_below_threshold_raises(self):
        with self.assertRaises(UncertainValueError):
            check_sensitive(Verified(5, confidence=0.9))

    def test_confidence_threshold_configurable(self):
        self.assertEqual(check_sensitive(Verified(5, confidence=0.9), min_confidence=0.9), 5)

    def test_allowed_statuses(self):
        self.assertEqual(
            check_sensitive(Stale(5, confidence=0.99), allowed_statuses=(Status.STALE,)), 5
        )
        with self.assertRaises(UncertainValueError):
            check_sensitive(Stale(5), allowed_statuses=(Status.VERIFIED,))

    def test_raw_value_passes(self):
        self.assertEqual(check_sensitive(5), 5)


if __name__ == "__main__":
    unittest.main()
