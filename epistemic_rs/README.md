# epistemic_rs

Rust port of the epistemic runtime — the Python library lives in `epistamic/` (workspace root).

## Build & test

```bash
cargo test
```

## Core

```rust
use epistemic_rs::{Belief, Epistemic, RecommendationKind, Status, recommend_action};

fn main() {
    let b = Belief::new(vec![Epistemic::verified(5u32)]);
    let r = recommend_action(&b, &[], None);
    assert_eq!(r.kind, RecommendationKind::None);
}
```
