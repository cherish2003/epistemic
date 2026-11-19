# epistemic

Values that carry their own epistemic properties: provenance, confidence, validity and conflicts.

## Install

```bash
pip install .
```

## Core

```python
from epistemic import Verified, Estimated, check_sensitive

delivery_time = Verified(14, source="driver-app")
est_eta = Estimated(14.5)

eta = delivery_time + est_eta   # status: Estimated — never silently Verified

check_sensitive(delivery_time)   # -> 14
check_sensitive(est_eta)         # raises UncertainValueError
```

## Agent runtime

```python
from epistemic import Belief, Claim, Status, VerifiedBy, recommend_action

belief = Belief([
    Claim(100, source="api_a", status=Status.VERIFIED),
    Claim(85, source="api_b", status=Status.VERIFIED),
])
belief.has_unresolved_conflicts()   # True

r = recommend_action(belief)        # kind: "verify" — resolve via independent sources
```

Policies: `VerifiedBy`, `IndependentConfirmation`, `Fresh`, `NoUnresolvedConflicts`.
Runtime gates: `enforce`, `EpistemicActionGate`.
Dispatch: `Recommendation`, `recommend_action` — returns one of
`verify` / `revalidate` / `observe` / `ask_human` / `none`.

## Frameworks

LangGraph adapter:

```python
from epistemic.langgraph import epistemic_tool_node

node = epistemic_tool_node(tools)
```

OpenAI Agents SDK adapter:

```python
from epistemic.openai_agents import EpistemicTools

adapter = EpistemicTools(trusted_tools=["search_api"])
```
