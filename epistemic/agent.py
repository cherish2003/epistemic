from __future__ import annotations

import operator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Generic, Optional, Sequence, TypeVar

from .core import Epistemic, Status, Unknown, Verified, combine

T = TypeVar("T")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Claim(Epistemic[T]):
    __slots__ = ("kind", "observed_at")

    def __init__(self, value=None, *, kind="claim", observed_at=None, **kwargs):
        kwargs.setdefault("status", Status.ESTIMATED)
        kwargs.setdefault("confidence", 0.7)
        self.kind = kind
        self.observed_at = observed_at or _utcnow()
        super().__init__(value, **kwargs)


class Derivation(Epistemic[T]):
    __slots__ = ("op", "inputs")

    def __init__(self, value, op, inputs, **kwargs):
        inputs = list(inputs)
        if inputs and "status" not in kwargs:
            combined = combine(*inputs)
            kwargs["status"] = combined.status
            if "confidence" not in kwargs:
                kwargs["confidence"] = combined.confidence
            if "evidence" not in kwargs:
                kwargs["evidence"] = combined.evidence
        self.op = op
        self.inputs = inputs
        super().__init__(value, **kwargs)

    @classmethod
    def from_inputs(cls, value, op, inputs):
        return cls(value, op, inputs)


class Belief:
    __slots__ = ("claims",)

    def __init__(self, claims=()):
        self.claims = [c if isinstance(c, Epistemic) else Verified(c) for c in claims]

    def has_unresolved_conflicts(self):
        if len(self.claims) < 2:
            return False
        first = self.claims[0].value
        return any(
            c.status is Status.CONFLICTING or c.value != first
            for c in self.claims[1:]
        )

    def values(self):
        return [c.value for c in self.claims]

    def sources(self):
        return [c.source for c in self.claims if c.source is not None]

    def candidates(self):
        return list(self.claims)

    def best(self, *policies):
        allowed = [c for c in self.claims if all(p.satisfied(c) for p in policies)]
        if not allowed:
            return None
        return max(allowed, key=lambda c: c.confidence or 0.0)


def _claims_of(value):
    return value.claims if isinstance(value, Belief) else [value]


class Policy:
    def __init__(self, name, predicate, description=""):
        self.name = name
        self.predicate = predicate
        self.description = description

    def satisfied(self, value):
        return bool(self.predicate(value))

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name!r})"


class VerifiedBy(Policy):
    def __init__(self, sources, min_independent=1):
        if isinstance(sources, str):
            sources = [sources]
        self.sources = tuple(sources)
        self.min_independent = min_independent
        super().__init__("verified_by", self._predicate, f"verified by {list(self.sources)}")

    def _predicate(self, value):
        distinct = {
            c.source
            for c in _claims_of(value)
            if c.status is Status.VERIFIED and c.source in self.sources
        }
        return len(distinct) >= self.min_independent


class IndependentConfirmation(Policy):
    def __init__(self, sources=None, min_sources=2):
        self.sources = tuple(sources) if sources is not None else None
        self.min_sources = min_sources
        super().__init__("independent_confirmation", self._predicate)

    def _predicate(self, value):
        distinct = {
            c.source
            for c in _claims_of(value)
            if c.status is Status.VERIFIED
            and c.source is not None
            and (self.sources is None or c.source in self.sources)
        }
        return len(distinct) >= self.min_sources


class Fresh(Policy):
    def __init__(self, within, description=None):
        self.within = within
        super().__init__("fresh", self._predicate, description or f"fresh within {within}")

    def _predicate(self, value):
        if isinstance(value, Belief):
            claims = value.candidates()
            if not claims:
                return False
            value = max(claims, key=lambda c: c.confidence or 0.0)
        if value.expires_at is None:
            return False
        return (value.expires_at - _utcnow()) >= self.within


class NoUnresolvedConflicts(Policy):
    def __init__(self):
        super().__init__("no_unresolved_conflicts", self._predicate)

    def _predicate(self, value):
        if isinstance(value, Belief):
            return not value.has_unresolved_conflicts()
        return not value.alternatives and value.status is not Status.CONFLICTING


class PolicyViolationError(ValueError):
    def __init__(self, policy, value):
        self.policy = policy
        self.value = value
        super().__init__(f"Policy violated: {policy.name} on {value!r}")


def enforce(value, *policies):
    for policy in policies:
        if not policy.satisfied(value):
            raise PolicyViolationError(policy, value)
    return value


def wrap_tool_result(tool_name, result, **kwargs):
    kwargs.setdefault("status", Status.ESTIMATED)
    kwargs.setdefault("confidence", 0.7)
    return Claim(result, source=tool_name, kind="tool_result", **kwargs)


@dataclass(frozen=True)
class Recommendation:
    kind: str
    claim: Epistemic
    rationale: str
    missing: tuple = ()
    options: tuple = ()


def _all_satisfied(value, policies):
    try:
        enforce(value, *policies)
        return True
    except PolicyViolationError:
        return False


def recommend_action(value, *, policies=(), now=None):
    claims = value.claims if isinstance(value, Belief) else [value]
    if not claims:
        return Recommendation("ask_human", Unknown(), "no claims to reason over")
    best = value.best() if isinstance(value, Belief) else value
    if isinstance(value, Belief) and value.has_unresolved_conflicts():
        verified = {
            c.source
            for c in claims
            if c.status is Status.VERIFIED and c.source is not None
        }
        options = tuple(s for s in value.sources() if s not in verified)
        return Recommendation(
            "verify",
            best,
            "belief has unresolved conflicts; resolve via independent sources",
            options=options,
        )
    if any(c.status_at(now) is Status.STALE for c in claims):
        return Recommendation(
            "revalidate", best, "the claim is stale; re-fetch evidence"
        )
    if any(c.status is Status.UNKNOWN for c in claims):
        options = tuple(c.source for c in claims if c.source is not None)
        return Recommendation(
            "observe",
            best,
            "the value is unknown; observe the listed sources",
            options=options,
        )
    if not _all_satisfied(value, policies):
        return Recommendation(
            "ask_human",
            best,
            "policies block the value and no further evidence is available",
        )
    return Recommendation("none", best, "no action needed")
