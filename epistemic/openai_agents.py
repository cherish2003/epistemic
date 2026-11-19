from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Optional, Sequence

from agents import FunctionTool

from .agent import Belief, Status, wrap_tool_result
from .agent import enforce


class EpistemicTools:
    """Wraps OpenAI Agents SDK tools: attaches provenance and gates consequential actions."""

    def __init__(
        self,
        *,
        trusted_tools: Sequence[str] = (),
        tool_policies: Optional[dict] = None,
        action_gate=None,
    ):
        self.trusted_tools = set(trusted_tools)
        self.tool_policies = dict(tool_policies or {})
        self.action_gate = action_gate
        self.claims_by_call_id = {}
        self.beliefs = {}

    def wrap_tool(self, tool: FunctionTool) -> FunctionTool:
        name = tool.name
        original = tool.on_invoke_tool
        trusted = name in self.trusted_tools
        policies = self.tool_policies.get(name, ())

        async def wrapped(ctx, args_json):
            result = await original(ctx, args_json)
            claim = wrap_tool_result(
                name,
                result,
                status=Status.VERIFIED if trusted else Status.ESTIMATED,
            )
            call_id = getattr(ctx, "tool_call_id", None)
            if call_id is not None:
                self.claims_by_call_id[call_id] = claim
            self.beliefs.setdefault(name, Belief()).claims.append(claim)
            if policies:
                enforce(claim, *policies)
            if self.action_gate is not None:
                self.action_gate.check(name, claim)
            return result

        return replace(tool, on_invoke_tool=wrapped)

    def wrap_agent_tools(self, agent) -> None:
        agent.tools = [self.wrap_tool(t) for t in agent.tools]
