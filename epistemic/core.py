from __future__ import annotations

import operator
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Generic, Optional, Sequence, TypeVar

T = TypeVar("T")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Status(Enum):
    VERIFIED = "verified"
    STALE = "stale"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"
    CONFLICTING = "conflicting"

    @property
    def precedence(self) -> int:
        return _PRECEDENCE[self]

    @property
    def default_confidence(self) -> float:
        return _DEFAULT_CONFIDENCE[self]


_PRECEDENCE = {
    Status.VERIFIED: 0,
    Status.STALE: 1,
    Status.ESTIMATED: 2,
    Status.UNKNOWN: 3,
    Status.CONFLICTING: 4,
}

_DEFAULT_CONFIDENCE = {
    Status.VERIFIED: 1.0,
    Status.STALE: 0.8,
    Status.ESTIMATED: 0.5,
    Status.UNKNOWN: 0.0,
    Status.CONFLICTING: 0.0,
}


@dataclass(frozen=True)
class Evidence:
    source: str
    type: str = "observation"
    timestamp: Optional[datetime] = None
    description: Optional[str] = None


class Epistemic(Generic[T]):
    __slots__ = (
        "_value",
        "status",
        "confidence",
        "source",
        "expires_at",
        "evidence",
        "notes",
        "alternatives",
        "timestamp",
    )

    def __init__(
        self,
        value: Optional[T] = None,
        status: Status = Status.VERIFIED,
        *,
        confidence: Optional[float] = None,
        source: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        expiration: Optional[datetime] = None,
        evidence: Sequence[Evidence] = (),
        notes: Optional[str] = None,
        alternatives: Sequence[Epistemic[T]] = (),
        timestamp: Optional[datetime] = None,
    ):
        self._value = value
        self.status = Status(status)
        self.confidence = confidence if confidence is not None else self.status.default_confidence
        self.source = source
        self.expires_at = expires_at if expires_at is not None else expiration
        self.evidence = list(evidence)
        self.notes = notes
        self.alternatives = list(alternatives)
        self.timestamp = timestamp or _utcnow()

    @property
    def value(self) -> Optional[T]:
        return self._value

    def as_value(self) -> Optional[T]:
        return self._value

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        if self.expires_at is None:
            return False
        return (now or _utcnow()) >= self.expires_at

    def status_at(self, now: Optional[datetime] = None) -> Status:
        if self.is_expired(now) and self.status in (
            Status.VERIFIED,
            Status.STALE,
            Status.ESTIMATED,
        ):
            return Status.STALE
        return self.status

    def is_current(self, now: Optional[datetime] = None) -> bool:
        return not self.is_expired(now)

    def __repr__(self) -> str:
        if self.alternatives:
            return f"{type(self).__name__}(alternatives={self.alternatives!r})"
        if self.notes:
            return (
                f"{type(self).__name__}({self._value!r}, status={self.status.value}, "
                f"notes={self.notes!r})"
            )
        return (
            f"{type(self).__name__}({self._value!r}, status={self.status.value}, "
            f"confidence={self.confidence})"
        )

    def __add__(self, other):
        return combine(self, other, op=operator.add)

    def __radd__(self, other):
        return combine(other, self, op=operator.add)

    def __sub__(self, other):
        return combine(self, other, op=operator.sub)

    def __rsub__(self, other):
        return combine(other, self, op=operator.sub)

    def __mul__(self, other):
        return combine(self, other, op=operator.mul)

    def __rmul__(self, other):
        return combine(other, self, op=operator.mul)


class Verified(Epistemic[T]):
    def __init__(self, value: T = None, *, confidence: float = 1.0, **kwargs):
        super().__init__(value, Status.VERIFIED, confidence=confidence, **kwargs)


class Stale(Epistemic[T]):
    def __init__(self, value: T = None, *, confidence: float = 0.8, **kwargs):
        super().__init__(value, Status.STALE, confidence=confidence, **kwargs)


class Estimated(Epistemic[T]):
    def __init__(self, value: T = None, confidence: float = 0.5, **kwargs):
        super().__init__(value, Status.ESTIMATED, confidence=confidence, **kwargs)


class Unknown(Epistemic[T]):
    def __init__(self, value: Optional[T] = None, reason: Optional[str] = None, **kwargs):
        kwargs.pop("confidence", None)
        notes = kwargs.pop("notes", None) or reason
        super().__init__(value, Status.UNKNOWN, confidence=0.0, notes=notes, **kwargs)


class Conflicting(Epistemic[T]):
    def __init__(self, alternatives: Sequence[Epistemic[T]] = (), **kwargs):
        kwargs.pop("confidence", None)
        super().__init__(
            None, Status.CONFLICTING, confidence=0.0, alternatives=alternatives, **kwargs
        )


def _flatten_conflicts(vals):
    out = []
    for v in vals:
        if v.status is Status.CONFLICTING and v.alternatives:
            out.extend(_flatten_conflicts(v.alternatives))
        else:
            out.append(v)
    return out


def combine(*values, op=operator.add, now=None):
    vals = [v if isinstance(v, Epistemic) else Verified(v) for v in values]
    if not vals:
        raise ValueError("combine requires at least one value")
    statuses = [v.status_at(now) for v in vals]
    worst = max(statuses, key=lambda s: s.precedence)
    if worst is Status.CONFLICTING:
        return Conflicting(alternatives=_flatten_conflicts(vals))
    if worst is Status.UNKNOWN:
        return Unknown()
    try:
        combined_value = op(*[v.value for v in vals])
    except Exception:
        combined_value = None
    confidences = [v.confidence for v in vals if v.confidence is not None]
    confidence = min(confidences) if confidences else None
    notes = "; ".join(v.notes for v in vals if v.notes) or None
    return Epistemic(
        combined_value,
        worst,
        confidence=confidence,
        evidence=[e for v in vals for e in v.evidence],
        notes=notes,
    )


class UncertainValueError(ValueError):
    def __init__(
        self,
        value: Epistemic,
        status: Status,
        confidence: Optional[float],
        allowed_statuses,
        min_confidence: float,
    ):
        self.value = value
        self.status = status
        self.confidence = confidence
        self.allowed_statuses = allowed_statuses
        self.min_confidence = min_confidence
        allowed = ", ".join(s.value for s in allowed_statuses)
        super().__init__(
            f"Uncertain value rejected: status={status.value!r} "
            f"(allowed: {allowed}), confidence={confidence} "
            f"(required >= {min_confidence})"
        )


def check_sensitive(
    value,
    *,
    allowed_statuses=(Status.VERIFIED,),
    min_confidence: float = 0.95,
    now: Optional[datetime] = None,
):
    v = value if isinstance(value, Epistemic) else Verified(value)
    status = v.status_at(now)
    if status not in allowed_statuses:
        raise UncertainValueError(v, status, v.confidence, allowed_statuses, min_confidence)
    if v.confidence is not None and v.confidence < min_confidence:
        raise UncertainValueError(v, status, v.confidence, allowed_statuses, min_confidence)
    return v.value
