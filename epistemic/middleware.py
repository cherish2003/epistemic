from __future__ import annotations

from typing import Callable, TypeVar

from .agent import Claim, Policy, Status, Verified, enforce, wrap_tool_result

T = TypeVar("T")


def wrap_tool(fn: Callable, *, tool_name=None, policies=(), trusted=False, return_claim=True):
    name = tool_name or getattr(fn, "__name__", "tool")

    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        claim = wrap_tool_result(
            name, result, status=Status.VERIFIED if trusted else Status.ESTIMATED
        )
        if policies:
            enforce(claim, *policies)
        return claim if return_claim else claim.value

    wrapper.__name__ = name
    return wrapper


class EpistemicActionGate:
    def __init__(self):
        self._policies = {}

    def register(self, action_name, *policies):
        self._policies[action_name] = policies

    def policies_for(self, action_name):
        return self._policies.get(action_name, ())

    def check(self, action_name, value):
        policies = self._policies.get(action_name)
        if policies:
            enforce(value, *policies)
        return value
