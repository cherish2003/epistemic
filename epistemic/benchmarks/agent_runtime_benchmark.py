"""Benchmark: epistemic runtime against adversarial tool outputs.

Scenarios mirror the failure modes agents face with real tool outputs:
stale results, conflicting results, unknown results, and policy-gated actions.
"""

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from epistemic import (
    Belief,
    Claim,
    EpistemicActionGate,
    PolicyViolationError,
    Status,
    VerifiedBy,
    combine,
    recommend_action,
)

RESULTS = []


def check(name, condition):
    RESULTS.append((name, bool(condition)))


def scenario_stale():
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    b = Belief([Claim(5, status=Status.VERIFIED, expires_at=past)])
    check("stale claim -> revalidate", recommend_action(b).kind == "revalidate")


def scenario_conflicting():
    b = Belief([
        Claim(100, source="api_a", status=Status.VERIFIED),
        Claim(85, source="api_b", status=Status.VERIFIED),
    ])
    check("conflicting sources -> verify", recommend_action(b).kind == "verify")


def scenario_unknown():
    b = Belief([Claim(None, status=Status.UNKNOWN, source="search_api")])
    r = recommend_action(b)
    check("unknown -> observe", r.kind == "observe" and "search_api" in r.options)


def scenario_blocked():
    gate = EpistemicActionGate()
    gate.register("payment.execute", VerifiedBy("finance_system"))
    b = Belief([Claim(5, source="finance", status=Status.ESTIMATED)])
    try:
        gate.check("payment.execute", b)
        check("blocked action -> raises", False)
    except PolicyViolationError:
        check("blocked action -> raises", True)


def scenario_consistent():
    b = Belief([Claim(5, status=Status.VERIFIED, source="api")])
    check("consistent -> none", recommend_action(b).kind == "none")


def perf_combine():
    claims = [Claim(i, status=Status.VERIFIED) for i in range(1000)]
    t0 = time.perf_counter()
    out = combine(*claims, op=lambda *xs: sum(xs))
    dt = time.perf_counter() - t0
    check("combine 1000 claims under 0.1s", dt < 0.1 and out.value == sum(range(1000)))


def main():
    for fn in (
        scenario_stale,
        scenario_conflicting,
        scenario_unknown,
        scenario_blocked,
        scenario_consistent,
        perf_combine,
    ):
        fn()
    passed = sum(1 for _, ok in RESULTS if ok)
    for name, ok in RESULTS:
        print(("PASS" if ok else "FAIL"), name)
    print(f"{passed}/{len(RESULTS)} checks passed")


if __name__ == "__main__":
    main()
