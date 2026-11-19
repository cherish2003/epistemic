# epistemic — type system for uncertain values

Values that carry their own provenance, confidence, validity and conflicts.

Agents make decisions on information they don't fully control — stale tool results, conflicting API outputs, memories that degrade, third-party data. This library gives software a native way to represent how much you should trust a value instead of flattening everything to a single value.

## Repo layout

- `epistemic/` — Python package (framework-integrated runtime)
  - `core.py` — `Epistemic`, `Status`, `Evidence`, `combine`, `check_sensitive`
  - `agent.py` — `Claim`, `Derivation`, `Belief`, policies (`VerifiedBy`, `IndependentConfirmation`, `Fresh`, `NoUnresolvedConflicts`), `enforce`, `Recommendation`, `recommend_action`
  - `middleware.py` — `wrap_tool`, `EpistemicActionGate`
  - `langgraph.py` — LangGraph adapter (`epistemic_tool_node`)
  - `openai_agents.py` — OpenAI Agents SDK adapter (`EpistemicTools`)
  - `examples/quickstart.py`
  - `benchmarks/agent_runtime_benchmark.py`
- `epistemic_rs/` — Rust crate (structural enforcement — `Verified<T>`)
- `tests/` — Python tests
- `pyproject.toml`

## Install (Python)

```bash
pip install .
```

## Core — values carry their trust

```python
from epistemic import Verified, Estimated, check_sensitive

delivery_time = Verified(14, source="driver-app")
est_eta = Estimated(14.5)

eta = delivery_time + est_eta   # uncertainty propagates — never silently "verified"

check_sensitive(eta)            # raises UncertainValueError
check_sensitive(delivery_time)  # -> 14
```

Every wrapper — `Verified`, `Estimated`, `Unknown`, `Conflicting`, `Stale` — carries status, confidence, evidence, expiry; arithmetic combines with worst-status propagation.

## Agent runtime — Belief, dispatch

```python
from epistemic import Belief, Claim, Status, VerifiedBy, recommend_action

belief = Belief([
    Claim(100, source="api_a", status=Status.VERIFIED),
    Claim(85, source="api_b", status=Status.VERIFIED),
])
belief.has_unresolved_conflicts()   # True

r = recommend_action(belief)        # kind: "verify"
```

## `recommend_action` — what should the agent do next?

| kind | meaning |
|---|---|
| `verify` | unresolved conflicts — resolve via independent sources |
| `revalidate` | claim is stale — re-fetch evidence |
| `observe` | value is unknown — observe the listed sources |
| `ask_human` | blocked by policies, no further evidence |
| `none` | nothing further needed |

## Frameworks — adapters

LangGraph:

```python
from epistemic.langgraph import epistemic_tool_node

node = epistemic_tool_node(tools)
```

Every tool result becomes a `Claim` (source, status, confidence); per-tool `Belief`s accumulate; consequential actions gated by `enforce`/`EpistemicActionGate.check`.

OpenAI Agents SDK:

```python
from epistemic.openai_agents import EpistemicTools

adapter = EpistemicTools(trusted_tools=["search_api"], action_gate=gate)
```

## Middleware — action gates

```python
from epistemic import EpistemicActionGate, VerifiedBy

gate = EpistemicActionGate()
gate.register("payment.execute", VerifiedBy("finance_system"))
```

## Rust — structural enforcement

`epistemic_rs` is a Rust crate; `Verified<T>` is only obtainable via `verify()` on the base type — sensitive functions taking `Verified<T>` can't be passed unverified values.

## Tests

```bash
python3 -m unittest discover
```

- Python: 84 tests pass (3.15: 10 skipped without langgraph/SDK; 3.10: full 84)

```bash
cd epistemic_rs && cargo test
```

- Rust: 55 tests pass

## Benchmark

```bash
python3 epistemic/benchmarks/agent_runtime_benchmark.py
```

Adversarial tool outputs — **6/6 PASS**: stale → revalidate, conflicting sources → verify, unknown → observe, blocked action raises, consistent → none, combine(1000 claims) < 0.1s.

## Quickstart

`python3 epistemic/examples/quickstart.py` — prints `conflicts: True`, `recommend: verify`.

## License

MIT — see `epistemic/LICENSE` and `epistemic_rs/LICENSE`.

## Why

Agents browse, code, access databases, send emails, execute transactions and coordinate. Autonomy without a native way to represent information trustworthiness is a fundamental gap: agents retrieve and act on stale, contradictory or incorrect information with no signal. This library gives software that signal: values carry provenance, confidence, validity and conflicts; actions can be gated on evidence; agents know what to do next.
