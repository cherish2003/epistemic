from __future__ import annotations

from typing import Callable, Sequence

from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode

from .agent import Belief, Status, enforce, wrap_tool_result
from .middleware import EpistemicActionGate


def epistemic_tool_node(
    tools,
    *,
    trusted_tools=(),
    tool_policies=None,
    action_gate=None,
    **kwargs,
):
    trusted = set(trusted_tools)
    policies = dict(tool_policies or {})
    claims_by_id = {}
    beliefs = {}

    def wrap_tool_call(request, execute):
        name = request.tool_call["name"]
        message = execute(request)
        claim = wrap_tool_result(
            name,
            message.content,
            status=Status.VERIFIED if name in trusted else Status.ESTIMATED,
        )
        call_id = request.tool_call["id"]
        claims_by_id[call_id] = claim
        beliefs.setdefault(name, Belief()).claims.append(claim)
        if name in policies:
            enforce(claim, *policies[name])
        if action_gate is not None:
            action_gate.check(name, claim)
        return message

    node = ToolNode(tools, wrap_tool_call=wrap_tool_call, **kwargs)
    node.epistemic_claims = claims_by_id
    node.epistemic_beliefs = beliefs
    node.epistemic_action_gate = action_gate
    return node
